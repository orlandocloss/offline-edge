import time
import os
import threading
import queue
import argparse
import random
import logging
import cv2
from datetime import datetime
from pathlib import Path

# Import our refactored components
from main.record_video import VideoRecorder
from main.inference_from_video import VideoInferenceProcessor, process_video
from models.insect_tracker import InsectTracker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContinuousPipeline:
    def __init__(self, video_dir="recordings", recording_duration=300, sanity_video_percentage=10, 
                 device_id="pipeline", fps=15, resolution=(640, 640), 
                 confidence_threshold=0.35, detections_dir="detections", sanity_videos_dir="sanity_videos",
                 recording_start_hour=6, recording_end_hour=22):
        """
        Initialize the continuous pipeline using imported components.
        """
        self.video_dir = Path(video_dir)
        self.sanity_video_percentage = sanity_video_percentage
        self.device_id = device_id
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.recording_start_hour = recording_start_hour
        self.recording_end_hour = recording_end_hour
        
        # Local storage directories
        self.detections_dir = Path(detections_dir)
        self.sanity_videos_dir = Path(sanity_videos_dir)
        
        # Create directories if they don't exist
        self.detections_dir.mkdir(exist_ok=True, parents=True)
        self.sanity_videos_dir.mkdir(exist_ok=True, parents=True)
        
        # Threading controls
        self.stop_event = threading.Event()
        self.video_queue = queue.Queue()

        # --- Centralized State Management ---
        self.global_frame_count = 0
        self.tracker = None
        
        # --- Centralized Tracker Initialization ---
        logger.info("Initializing insect tracker...")
        try:
            width, height = self.resolution
            self.tracker = InsectTracker(height, width, max_frames=30, w_dist=0.7, w_area=0.3, cost_threshold=0.8)
            logger.info("✅ Insect tracker initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize insect tracker: {e}. Exiting.")
            raise
        
        # Initialize components
        self.recorder = VideoRecorder(
            output_dir=str(self.video_dir),
            fps=fps,
            resolution=resolution,
            recording_duration=recording_duration,
            device_id=self.device_id,
            video_queue=self.video_queue,
            recording_start_hour=self.recording_start_hour,
            recording_end_hour=self.recording_end_hour
        )
        
        # Threads
        self.recorder_thread = None
        self.processor_thread = None
        
        logger.info("🚀 Continuous Pipeline initialized")
        logger.info(f"📁 Detections directory: {self.detections_dir}")
        logger.info(f"📁 Sanity videos directory: {self.sanity_videos_dir}")
        logger.info(f"⏰ Recording schedule: {self.recording_start_hour:02d}:00 - {self.recording_end_hour:02d}:00")
        logger.info("ℹ️  Video processing will continue 24/7, but new recordings only during scheduled hours")

    def processor_worker(self):
        """Worker thread to process videos from the queue."""
        logger.info("🔍 Starting video processor worker...")
        
        while not self.stop_event.is_set() or not self.video_queue.empty():
            try:
                video_path = self.video_queue.get(timeout=1)
                
                logger.info(f"--- Processing Video: {video_path.name} ---")
                
                self.run_inference_on_video(video_path)
                self.save_sanity_video(video_path)
                self.delete_video(video_path)
                
                self.video_queue.task_done()
                
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue
            except Exception as e:
                logger.error(f"❌ Processor worker error: {e}", exc_info=True)
        
        logger.info("🔍 Video processor worker stopped")

    def run_inference_on_video(self, video_path):
        """Run the full inference process on a video file."""
        logger.info(f"🧠 Running inference on {video_path.name}")
        
        # Log tracking state before processing
        if self.tracker:
            stats = self.tracker.get_tracking_stats()
            logger.info(f"📊 Tracker state before {video_path.name}: {stats['active_tracks']} active, {stats['lost_tracks']} lost tracks")
            if stats['active_tracks'] > 0:
                logger.info(f"🔗 Cross-video tracking: Continuing with existing tracks: {stats['active_track_ids'][:3]}{'...' if len(stats['active_track_ids']) > 3 else ''}")
        
        try:
            processor = VideoInferenceProcessor(
                detections_dir=str(self.detections_dir),
                sanity_videos_dir=str(self.sanity_videos_dir),
                device_id=self.device_id,
                confidence_threshold=self.confidence_threshold
            )
            frames_processed = process_video(
                video_path=str(video_path), 
                processor=processor,
                tracker=self.tracker,
                start_frame_count=self.global_frame_count
            )
            self.global_frame_count += frames_processed
            
            # Log tracking state after processing
            if self.tracker:
                stats_after = self.tracker.get_tracking_stats()
                logger.info(f"📊 Tracker state after {video_path.name}: {stats_after['active_tracks']} active, {stats_after['lost_tracks']} lost tracks")

        except Exception as e:
            logger.error(f"Error during inference for {video_path.name}: {e}")

    def should_save_sanity_video(self):
        """Determine if this video should be saved as a sanity video based on percentage."""
        return random.randint(1, 100) <= self.sanity_video_percentage

    def save_sanity_video(self, video_path):
        """
        Save a video segment as a sanity video.
        The duration of the segment is determined by self.sanity_video_percentage.
        """
        if self.sanity_video_percentage == 0:
            return
        
        if not self.should_save_sanity_video():
            logger.info(f"🎯 Skipping sanity video for {video_path.name} (random selection)")
            return
        
        if not (0 < self.sanity_video_percentage <= 100):
            logger.error(f"Invalid sanity video percentage: {self.sanity_video_percentage}. Must be between 1 and 100.")
            return

        cap = None
        out = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video file to create sanity video: {video_path.name}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames == 0 or fps == 0:
                logger.error(f"Video {video_path.name} has no frames or invalid FPS, cannot create sanity video.")
                return

            # Calculate segment length in frames
            segment_length_frames = int(total_frames * (self.sanity_video_percentage / 100.0))
            if segment_length_frames <= 0:
                logger.warning(f"Calculated segment length is 0 frames for {video_path.name}. Skipping sanity video.")
                return

            # Determine random start frame
            max_start_frame = total_frames - segment_length_frames
            start_frame = random.randint(0, max_start_frame) if max_start_frame > 0 else 0

            # Create sanity video path
            sanity_video_path = self.sanity_videos_dir / f"sanity_{video_path.name}"

            logger.info(f"🎬 Creating a {self.sanity_video_percentage}% ({segment_length_frames / fps:.1f}s) sanity video from {video_path.name}")
            
            # Set up writer for the sanity video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(sanity_video_path), fourcc, fps, (width, height))

            # Position the capture to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_written = 0
            while frames_written < segment_length_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written += 1

            logger.info(f"✅ Sanity video saved: {sanity_video_path.name}")
            
        except Exception as e:
            logger.error(f"❌ Error creating sanity video for {video_path.name}: {e}", exc_info=True)
        finally:
            if cap:
                cap.release()
            if out:
                out.release()
            
    def delete_video(self, video_path):
        """Delete a video file after processing."""
        try:
            video_path.unlink()
            logger.info(f"🗑️ Deleted video: {video_path.name}")
        except Exception as e:
            logger.error(f"⚠️ Could not delete video {video_path.name}: {e}")

    def start(self):
        """Start the continuous pipeline."""
        logger.info("🚀 Starting Continuous Pipeline...")
        
        # Scan for and queue any existing videos from previous runs
        logger.info(f"Scanning for existing videos in {self.video_dir}...")
        try:
            existing_videos = sorted(self.video_dir.glob("*.mp4"))
            if existing_videos:
                logger.info(f"Found {len(existing_videos)} existing video(s). Adding to processing queue.")
                for video_path in existing_videos:
                    self.video_queue.put(video_path)
            else:
                logger.info("No existing videos found.")
        except Exception as e:
            logger.error(f"Error scanning for existing videos: {e}")

        self.recorder_thread = threading.Thread(target=self.recorder.start_continuous_recording, daemon=True)
        self.processor_thread = threading.Thread(target=self.processor_worker, daemon=True)
        self.recorder_thread.start()
        self.processor_thread.start()
        logger.info("✅ All worker threads started")
        
    def stop(self):
        """Stop the continuous pipeline gracefully, ensuring all work is finished."""
        logger.info("🔄 Stopping pipeline... recorder will finish current segment.")
        
        # 1. Signal recorder to stop. It will finish its current segment and push it to the queue.
        self.recorder.stop()
        # 2. Wait for the recorder thread to finish its job.
        if self.recorder_thread and self.recorder_thread.is_alive():
            self.recorder_thread.join()
        logger.info("✅ Recorder thread has stopped.")
        
        # 3. Wait for the processor to clear the queue.
        logger.info("...waiting for processor to finish all remaining videos...")
        self.video_queue.join()
        
        # 4. Signal the processor worker to stop now that the queue is empty.
        self.stop_event.set()
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join()
        logger.info("✅ Processor thread has stopped.")
        
        logger.info("✅ Pipeline stopped gracefully.")

def main():
    parser = argparse.ArgumentParser(description='Continuous video recording and inference pipeline.')
    parser.add_argument('--video-dir', type=str, default='recordings', help='Directory to save and monitor videos.')
    parser.add_argument('--duration', type=int, default=300, help='Duration of each video segment in seconds.')
    parser.add_argument('--sanity-video-percentage', type=int, default=10, help='Percentage of video *duration* to save as sanity video segment (0-100).')
    parser.add_argument('--device-id', type=str, default='pipeline', help='Device identifier for detections.')
    parser.add_argument('--fps', type=int, default=15, help='Recording frame rate.')
    parser.add_argument('--resolution', type=str, default='640x640', help='Recording resolution in WIDTHxHEIGHT format.')
    parser.add_argument('--confidence', type=float, default=0.35, help='Confidence threshold for detections.')
    parser.add_argument('--detections-dir', type=str, default='detections', help='Directory to save detection frames and data.')
    parser.add_argument('--sanity-videos-dir', type=str, default='sanity_videos', help='Directory to save sanity video segments.')
    parser.add_argument('--recording-start-hour', type=int, default=6, help='Hour to start recording (24-hour format, default 6 = 6am).')
    parser.add_argument('--recording-end-hour', type=int, default=22, help='Hour to stop recording (24-hour format, default 22 = 10pm).')
    
    args = parser.parse_args()
    
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        logger.error(f"Error: Invalid resolution format '{args.resolution}'. Use WIDTHxHEIGHT.")
        return 1
    
    # Validate recording hours
    if not (0 <= args.recording_start_hour <= 23) or not (0 <= args.recording_end_hour <= 23):
        logger.error(f"Error: Recording hours must be between 0 and 23. Got start={args.recording_start_hour}, end={args.recording_end_hour}")
        return 1
    
    if args.recording_start_hour >= args.recording_end_hour:
        logger.error(f"Error: Recording start hour ({args.recording_start_hour}) must be less than end hour ({args.recording_end_hour})")
        return 1
    
    # Validate sanity video percentage
    if not (0 <= args.sanity_video_percentage <= 100):
        logger.error(f"Error: Sanity video percentage must be between 0 and 100. Got {args.sanity_video_percentage}")
        return 1
    
    pipeline = ContinuousPipeline(
        video_dir=args.video_dir,
        recording_duration=args.duration,
        sanity_video_percentage=args.sanity_video_percentage,
        device_id=args.device_id,
        fps=args.fps,
        resolution=resolution,
        confidence_threshold=args.confidence,
        detections_dir=args.detections_dir,
        sanity_videos_dir=args.sanity_videos_dir,
        recording_start_hour=args.recording_start_hour,
        recording_end_hour=args.recording_end_hour
    )
    
    try:
        pipeline.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown initiated by user.")
    except Exception as e:
        logger.error(f"💥 Fatal error in main loop: {e}", exc_info=True)
    finally:
        pipeline.stop()
    
    return 0

if __name__ == "__main__":
    exit(main())
