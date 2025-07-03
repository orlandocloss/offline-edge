import time
import os
import threading
import queue
import argparse
import random
import logging
import cv2
import sys
from datetime import datetime
from pathlib import Path

# Import our refactored components
from main.record_video import VideoRecorder
from main.inference_from_video import VideoInferenceProcessor, process_video
from models.insect_tracker import InsectTracker

# Set up logging with immediate flushing for SSH
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force immediate output flushing
def flush_print(msg):
    print(msg)
    sys.stdout.flush()

class ContinuousPipeline:
    def __init__(self, video_dir="recordings", recording_duration=300, sanity_video_percentage=10, 
                 device_id="pipeline", fps=15, resolution=(640, 640), 
                 confidence_threshold=0.35, detections_dir="detections", sanity_videos_dir="sanity_videos",
                 recording_start_hour=6, recording_end_hour=22, verbose=False):
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
        self.verbose = verbose
        
        # Local storage directories
        self.detections_dir = Path(detections_dir)
        self.sanity_videos_dir = Path(sanity_videos_dir)
        
        # Create directories if they don't exist
        self.detections_dir.mkdir(exist_ok=True, parents=True)
        self.sanity_videos_dir.mkdir(exist_ok=True, parents=True)
        
        flush_print(f"✅ Created directories: {self.detections_dir}, {self.sanity_videos_dir}")
        
        # Threading controls
        self.stop_event = threading.Event()
        self.video_queue = queue.Queue()

        # --- Centralized State Management ---
        self.global_frame_count = 0
        self.tracker = None
        
        # --- Centralized Tracker Initialization ---
        flush_print("🔄 Initializing insect tracker...")
        try:
            width, height = self.resolution
            self.tracker = InsectTracker(height, width, max_frames=30, w_dist=0.7, w_area=0.3, cost_threshold=0.8)
            flush_print("✅ Insect tracker initialized successfully.")
        except Exception as e:
            flush_print(f"❌ Failed to initialize insect tracker: {e}")
            raise
        
        # Initialize components
        flush_print("🔄 Initializing video recorder...")
        self.recorder = VideoRecorder(
            output_dir=str(self.video_dir),
            fps=fps,
            resolution=resolution,
            recording_duration=recording_duration,
            device_id=self.device_id,
            video_queue=self.video_queue,
            recording_start_hour=self.recording_start_hour,
            recording_end_hour=self.recording_end_hour,
            verbose=self.verbose
        )
        flush_print("✅ Video recorder initialized.")
        
        # Threads
        self.recorder_thread = None
        self.processor_thread = None
        
        flush_print("🚀 Continuous Pipeline initialized")
        flush_print(f"📁 Detections directory: {self.detections_dir}")
        flush_print(f"📁 Sanity videos directory: {self.sanity_videos_dir}")
        flush_print(f"⏰ Recording schedule: {self.recording_start_hour:02d}:00 - {self.recording_end_hour:02d}:00")
        flush_print("ℹ️  Video processing will continue 24/7, but new recordings only during scheduled hours")

    def processor_worker(self):
        """Worker thread to process videos from the queue."""
        flush_print("🔍 Starting video processor worker...")
        
        while not self.stop_event.is_set() or not self.video_queue.empty():
            try:
                if self.verbose:
                    flush_print(f"📋 Queue size: {self.video_queue.qsize()}, Stop event: {self.stop_event.is_set()}")
                
                video_path = self.video_queue.get(timeout=1)
                
                flush_print(f"--- Processing Video: {video_path.name} ---")
                
                self.run_inference_on_video(video_path)
                self.save_sanity_video(video_path)
                self.delete_video(video_path)
                
                self.video_queue.task_done()
                flush_print(f"✅ Finished processing: {video_path.name}")
                
            except queue.Empty:
                if self.verbose:
                    flush_print("⏳ Queue empty, checking stop event...")
                if self.stop_event.is_set():
                    flush_print("🛑 Stop event set, breaking from processor loop")
                    break
                continue
            except Exception as e:
                flush_print(f"❌ Processor worker error: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
        
        flush_print("🔍 Video processor worker stopped")

    def run_inference_on_video(self, video_path):
        """Run the full inference process on a video file."""
        flush_print(f"🧠 Running inference on {video_path.name}")
        
        # Log tracking state before processing
        if self.tracker:
            stats = self.tracker.get_tracking_stats()
            flush_print(f"📊 Tracker state before {video_path.name}: {stats['active_tracks']} active, {stats['lost_tracks']} lost tracks")
            if stats['active_tracks'] > 0:
                flush_print(f"🔗 Cross-video tracking: Continuing with existing tracks: {stats['active_track_ids'][:3]}{'...' if len(stats['active_track_ids']) > 3 else ''}")
        
        try:
            processor = VideoInferenceProcessor(
                detections_dir=str(self.detections_dir),
                sanity_videos_dir=str(self.sanity_videos_dir),
                device_id=self.device_id,
                confidence_threshold=self.confidence_threshold,
                verbose=self.verbose
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
                flush_print(f"📊 Tracker state after {video_path.name}: {stats_after['active_tracks']} active, {stats_after['lost_tracks']} lost tracks")

        except Exception as e:
            flush_print(f"❌ Error during inference for {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    def save_sanity_video(self, video_path):
        """
        Save a video segment as a sanity video.
        The duration of the segment is determined by self.sanity_video_percentage.
        """
        flush_print(f"🎬 Creating sanity video for {video_path.name}...")
        
        if self.sanity_video_percentage == 0:
            flush_print(f"⏭️  Sanity video disabled (percentage = 0)")
            return
        
        if not (0 < self.sanity_video_percentage <= 100):
            flush_print(f"❌ Invalid sanity video percentage: {self.sanity_video_percentage}. Must be between 1 and 100.")
            return

        cap = None
        out = None
        try:
            flush_print(f"📹 Opening video file: {video_path}")
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                flush_print(f"❌ Could not open video file to create sanity video: {video_path.name}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_duration = total_frames / fps if fps > 0 else 0

            flush_print(f"📊 Video properties: {total_frames} frames, {fps:.2f} FPS, {width}x{height}, {total_duration:.1f}s")

            if total_frames == 0 or fps == 0:
                flush_print(f"❌ Video {video_path.name} has no frames or invalid FPS, cannot create sanity video.")
                return

            # Calculate segment length in frames (percentage of total duration)
            segment_length_frames = int(total_frames * (self.sanity_video_percentage / 100.0))
            segment_duration = segment_length_frames / fps if fps > 0 else 0
            
            if segment_length_frames <= 0:
                flush_print(f"⚠️ Calculated segment length is 0 frames for {video_path.name}. Skipping sanity video.")
                return

            # Determine random start frame
            max_start_frame = total_frames - segment_length_frames
            start_frame = random.randint(0, max_start_frame) if max_start_frame > 0 else 0
            start_time = start_frame / fps if fps > 0 else 0

            # Create sanity video path
            sanity_video_path = self.sanity_videos_dir / f"sanity_{video_path.name}"

            flush_print(f"🎬 Extracting {self.sanity_video_percentage}% ({segment_duration:.1f}s) from {video_path.name}")
            flush_print(f"📂 Sanity video will be saved to: {sanity_video_path}")
            flush_print(f"🎯 Random segment: {start_time:.1f}s to {start_time + segment_duration:.1f}s (frames {start_frame} to {start_frame + segment_length_frames})")
            
            # Set up writer for the sanity video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(sanity_video_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                flush_print(f"❌ Failed to open video writer for {sanity_video_path}")
                return

            # Position the capture to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_written = 0
            while frames_written < segment_length_frames:
                ret, frame = cap.read()
                if not ret:
                    flush_print(f"⚠️ Could not read frame {start_frame + frames_written}, stopping at {frames_written} frames")
                    break
                out.write(frame)
                frames_written += 1
                
                # Progress update for longer segments
                if self.verbose and frames_written % 100 == 0:
                    progress = (frames_written / segment_length_frames) * 100
                    flush_print(f"📹 Sanity video progress: {frames_written}/{segment_length_frames} frames ({progress:.1f}%)")

            # Check if file was actually created and has content
            file_size = 0
            if sanity_video_path.exists():
                file_size = sanity_video_path.stat().st_size / (1024 * 1024)  # MB
            
            if file_size > 0:
                flush_print(f"✅ Sanity video saved: {sanity_video_path.name} ({frames_written} frames, {file_size:.1f} MB)")
            else:
                flush_print(f"❌ Sanity video file is empty or was not created: {sanity_video_path.name}")
            
        except Exception as e:
            flush_print(f"❌ Error creating sanity video for {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        finally:
            if cap:
                cap.release()
            if out:
                out.release()
            
    def delete_video(self, video_path):
        """Delete a video file after processing."""
        try:
            video_path.unlink()
            flush_print(f"🗑️ Deleted video: {video_path.name}")
        except Exception as e:
            flush_print(f"⚠️ Could not delete video {video_path.name}: {e}")

    def start(self):
        """Start the continuous pipeline."""
        flush_print("🚀 Starting Continuous Pipeline...")
        
        # Scan for and queue any existing videos from previous runs
        flush_print(f"📂 Scanning for existing videos in {self.video_dir}...")
        try:
            existing_videos = sorted(self.video_dir.glob("*.mp4"))
            if existing_videos:
                flush_print(f"📹 Found {len(existing_videos)} existing video(s). Adding to processing queue.")
                for video_path in existing_videos:
                    self.video_queue.put(video_path)
                    flush_print(f"  ➕ Queued: {video_path.name}")
            else:
                flush_print("📂 No existing videos found.")
        except Exception as e:
            flush_print(f"❌ Error scanning for existing videos: {e}")

        flush_print("🎯 Starting recorder thread...")
        self.recorder_thread = threading.Thread(target=self.recorder.start_continuous_recording, daemon=True)
        self.recorder_thread.start()
        
        flush_print("🎯 Starting processor thread...")
        self.processor_thread = threading.Thread(target=self.processor_worker, daemon=True)
        self.processor_thread.start()
        
        flush_print("✅ All worker threads started")
        flush_print("🔄 Pipeline is now running. Press Ctrl+C to stop...")
        
        # Add a heartbeat to show the main thread is alive
        heartbeat_count = 0
        while True:
            try:
                time.sleep(30)  # Heartbeat every 30 seconds
                heartbeat_count += 1
                if self.verbose or heartbeat_count % 2 == 0:  # Show every minute, or every 30s if verbose
                    flush_print(f"💓 Heartbeat {heartbeat_count}: Pipeline running (Queue: {self.video_queue.qsize()}, Recorder: {'alive' if self.recorder_thread.is_alive() else 'dead'}, Processor: {'alive' if self.processor_thread.is_alive() else 'dead'})")
            except KeyboardInterrupt:
                flush_print("🛑 Keyboard interrupt received")
                break
            except Exception as e:
                flush_print(f"⚠️ Error in main loop: {e}")
                break
        
    def stop(self):
        """Stop the continuous pipeline gracefully, ensuring all work is finished."""
        flush_print("🔄 Stopping pipeline... recorder will finish current segment.")
        
        # 1. Signal recorder to stop. It will finish its current segment and push it to the queue.
        self.recorder.stop()
        # 2. Wait for the recorder thread to finish its job.
        if self.recorder_thread and self.recorder_thread.is_alive():
            flush_print("⏳ Waiting for recorder thread to finish...")
            self.recorder_thread.join()
        flush_print("✅ Recorder thread has stopped.")
        
        # 3. Wait for the processor to clear the queue.
        flush_print("⏳ Waiting for processor to finish all remaining videos...")
        self.video_queue.join()
        
        # 4. Signal the processor worker to stop now that the queue is empty.
        self.stop_event.set()
        if self.processor_thread and self.processor_thread.is_alive():
            flush_print("⏳ Waiting for processor thread to finish...")
            self.processor_thread.join()
        flush_print("✅ Processor thread has stopped.")
        
        flush_print("✅ Pipeline stopped gracefully.")

def main():
    parser = argparse.ArgumentParser(description='Continuous video recording and inference pipeline.')
    parser.add_argument('--video-dir', type=str, default='recordings', help='Directory to save and monitor videos.')
    parser.add_argument('--duration', type=int, default=300, help='Duration of each video segment in seconds.')
    parser.add_argument('--sanity-video-percentage', type=int, default=10, help='Percentage of each video duration to extract as sanity video. 10% = extract 30s from a 300s video (0-100).')
    parser.add_argument('--device-id', type=str, default='pipeline', help='Device identifier for detections.')
    parser.add_argument('--fps', type=int, default=15, help='Recording frame rate.')
    parser.add_argument('--resolution', type=str, default='640x640', help='Recording resolution in WIDTHxHEIGHT format.')
    parser.add_argument('--confidence', type=float, default=0.35, help='Confidence threshold for detections.')
    parser.add_argument('--detections-dir', type=str, default='detections', help='Directory to save detection frames and data.')
    parser.add_argument('--sanity-videos-dir', type=str, default='sanity_videos', help='Directory to save sanity video segments.')
    parser.add_argument('--recording-start-hour', type=int, default=6, help='Hour to start recording (24-hour format, default 6 = 6am).')
    parser.add_argument('--recording-end-hour', type=int, default=22, help='Hour to stop recording (24-hour format, default 22 = 10pm).')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose debug output.')
    
    args = parser.parse_args()
    
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        flush_print(f"❌ Error: Invalid resolution format '{args.resolution}'. Use WIDTHxHEIGHT.")
        return 1
    
    # Validate recording hours
    if not (0 <= args.recording_start_hour <= 23) or not (0 <= args.recording_end_hour <= 23):
        flush_print(f"❌ Error: Recording hours must be between 0 and 23. Got start={args.recording_start_hour}, end={args.recording_end_hour}")
        return 1
    
    if args.recording_start_hour >= args.recording_end_hour:
        flush_print(f"❌ Error: Recording start hour ({args.recording_start_hour}) must be less than end hour ({args.recording_end_hour})")
        return 1
    
    # Validate sanity video percentage
    if not (0 <= args.sanity_video_percentage <= 100):
        flush_print(f"❌ Error: Sanity video percentage must be between 0 and 100. Got {args.sanity_video_percentage}")
        return 1
    
    flush_print("🚀 Initializing pipeline...")
    
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
        recording_end_hour=args.recording_end_hour,
        verbose=args.verbose
    )
    
    try:
        pipeline.start()
    except KeyboardInterrupt:
        flush_print("🛑 Shutdown initiated by user.")
    except Exception as e:
        flush_print(f"💥 Fatal error in main loop: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        pipeline.stop()
    
    return 0

if __name__ == "__main__":
    exit(main())
