import cv2
import time
import os
import threading
import queue
import logging
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
from picamera2 import Picamera2

logger = logging.getLogger(__name__)

# Force immediate output flushing
def flush_print(msg):
    print(msg)
    sys.stdout.flush()

class VideoRecorder:
    def __init__(self, output_dir="recordings", fps=15, resolution=(640, 640), 
                 recording_duration=300, device_id="recorder", video_queue=None,
                 recording_start_hour=6, recording_end_hour=22, verbose=False):
        """
        Initialize the video recorder for gapless recording.
        
        Args:
            output_dir: Directory to save video files
            fps: Frames per second for recording
            resolution: Video resolution (width, height)
            recording_duration: Duration of each video segment in seconds
            device_id: Device identifier for filename
            video_queue: A queue to put the finished video file paths into.
            recording_start_hour: Hour to start recording (24-hour format, default 6 = 6am)
            recording_end_hour: Hour to stop recording (24-hour format, default 22 = 10pm)
            verbose: Enable verbose debug output
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.resolution = resolution
        self.recording_duration = recording_duration
        self.device_id = device_id
        self.video_queue = video_queue
        self.recording_start_hour = recording_start_hour
        self.recording_end_hour = recording_end_hour
        self.verbose = verbose
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.picam2 = None
        self.stop_event = threading.Event()
        
        # Frame queue for decoupling capture from writing
        self.frame_queue = queue.Queue(maxsize=fps * 5) # Buffer 5 seconds of frames
        self.frame_grabber_thread = None
        
        flush_print("✅ Video Recorder component initialized.")
        flush_print(f"  - FPS: {self.fps}, Resolution: {self.resolution}")
        flush_print(f"  - Recording schedule: {self.recording_start_hour:02d}:00 - {self.recording_end_hour:02d}:00")
    
    def is_recording_time(self):
        """Check if current time is within recording hours."""
        current_hour = datetime.now().hour
        return self.recording_start_hour <= current_hour < self.recording_end_hour
    
    def initialize_camera(self):
        """Initialize and configure the camera."""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                flush_print(f"📸 Initializing camera (attempt {attempt + 1}/{max_retries})...")
                
                # Clean up any existing camera instance
                if self.picam2:
                    try:
                        self.picam2.stop()
                        self.picam2 = None
                    except:
                        pass
                
                self.picam2 = Picamera2()
                camera_config = self.picam2.create_video_configuration(
                    main={"format": 'RGB888', "size": self.resolution}
                )
                self.picam2.configure(camera_config)
                self.picam2.set_controls({
                    "FrameRate": float(self.fps),
                    "AfMode": 0, "LensPosition": 0.0,
                })
                self.picam2.start()
                flush_print("⏳ Camera initializing (waiting 2 seconds)...")
                time.sleep(2)
                flush_print("✅ Camera ready.")
                return  # Success, exit retry loop
                
            except Exception as e:
                flush_print(f"❌ Camera initialization attempt {attempt + 1} failed: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                
                if attempt < max_retries - 1:
                    flush_print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    flush_print("❌ All camera initialization attempts failed!")
                    raise

    def _stop_camera(self):
        """Stop the camera and frame grabber thread."""
        if self.verbose:
            flush_print("🔄 Stopping camera and frame grabber...")
        
        self.stop_event.set()
        
        if self.frame_grabber_thread and self.frame_grabber_thread.is_alive():
            if self.verbose:
                flush_print("⏳ Waiting for frame grabber thread to stop...")
            self.frame_grabber_thread.join(timeout=2)
            
        if self.picam2:
            try:
                self.picam2.stop()
                flush_print("📸 Camera stopped for time restriction")
            except Exception as e:
                flush_print(f"⚠️ Warning: Error stopping camera: {e}")
        
        # Clear the stop event so recording can resume later
        self.stop_event.clear()
        if self.verbose:
            flush_print("✅ Camera and frame grabber stopped")

    def _frame_grabber(self):
        """Continuously grabs frames from the camera and puts them in a queue."""
        flush_print("📹 Frame grabber thread started.")
        frame_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while not self.stop_event.is_set():
            try:
                frame = self.picam2.capture_array()
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                    frame_count += 1
                    consecutive_errors = 0  # Reset error counter on success
                    
                    if self.verbose and frame_count % 100 == 0:
                        flush_print(f"📹 Frame grabber: {frame_count} frames captured, queue size: {self.frame_queue.qsize()}")
                except queue.Full:
                    if self.verbose:
                        flush_print("⚠️ Frame queue full, dropping frame")
                    continue
                    
            except Exception as e:
                consecutive_errors += 1
                if not self.stop_event.is_set():
                    flush_print(f"❌ Error in frame grabber thread ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        flush_print("❌ Too many consecutive frame grabber errors, stopping thread")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                        break
                    
                    # Brief pause before retrying
                    time.sleep(0.1)
                else:
                    break
                    
        flush_print("📹 Frame grabber thread stopped.")

    def record_segment(self, output_path):
        """Record a single video segment by consuming frames from the queue."""
        flush_print(f"🎬 Recording new segment: {output_path.name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, self.resolution)
        
        if not out.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        start_time = time.time()
        frame_count = 0
        last_progress_time = start_time
        
        try:
            while time.time() - start_time < self.recording_duration:
                try:
                    frame = self.frame_queue.get(timeout=1)
                    out.write(frame)
                    frame_count += 1
                    
                    # Progress update every 30 seconds or if verbose every 10 seconds
                    current_time = time.time()
                    progress_interval = 10 if self.verbose else 30
                    if current_time - last_progress_time >= progress_interval:
                        elapsed = current_time - start_time
                        remaining = self.recording_duration - elapsed
                        progress = (elapsed / self.recording_duration) * 100
                        flush_print(f"🎬 Recording progress: {progress:.1f}% ({elapsed:.0f}s/{self.recording_duration}s), {frame_count} frames, queue: {self.frame_queue.qsize()}")
                        last_progress_time = current_time
                    
                except queue.Empty:
                    if self.stop_event.is_set():
                        flush_print("🛑 Stop event received, finishing current segment early.")
                        break
                    if self.verbose:
                        flush_print("⏳ No frames available, waiting...")
                    continue
        finally:
            out.release()
            
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        recording_time = time.time() - start_time
        actual_fps = frame_count / recording_time if recording_time > 0 else 0
        flush_print(f"✅ Segment complete: {output_path.name} ({frame_count} frames, {file_size_mb:.1f} MB, {actual_fps:.1f} FPS)")
        
        if self.video_queue:
            self.video_queue.put(output_path)
            flush_print(f"📤 Added to processing queue: {output_path.name}")
    
    def start_continuous_recording(self):
        """Start continuous recording loop. This is intended to be the target of a thread."""
        flush_print("🚀 Starting continuous recording loop...")
        camera_initialized = False
        
        try:
            while not self.stop_event.is_set():
                current_time = datetime.now()
                current_hour = current_time.hour
                
                if self.is_recording_time():
                    # Initialize camera only when we need to record
                    if not camera_initialized:
                        flush_print(f"⏰ Entered recording hours (current time: {current_time.strftime('%H:%M:%S')}) - initializing camera...")
                        try:
                            self.initialize_camera()
                            self.frame_grabber_thread = threading.Thread(target=self._frame_grabber, daemon=True)
                            self.frame_grabber_thread.start()
                            camera_initialized = True
                        except Exception as e:
                            flush_print(f"❌ Failed to initialize camera during recording hours: {e}")
                            flush_print("⏳ Will retry camera initialization in 5 minutes...")
                            time.sleep(300)  # Wait 5 minutes before retrying
                            continue
                    
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.device_id}_{timestamp}.mp4"
                    output_path = self.output_dir / filename
                    
                    try:
                        self.record_segment(output_path)
                    except Exception as e:
                        flush_print(f"❌ Error recording segment: {e}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                        
                        # Check if this is a camera-related error
                        if "camera" in str(e).lower() or "frame" in str(e).lower():
                            flush_print("❌ Camera error detected, reinitializing camera...")
                            try:
                                self._stop_camera()
                                camera_initialized = False
                                time.sleep(30)  # Wait 30 seconds before retrying
                            except Exception as recovery_error:
                                flush_print(f"❌ Error during camera recovery: {recovery_error}")
                        
                        # Continue to next segment
                        continue
                        
                else:
                    # Outside recording hours
                    if camera_initialized:
                        flush_print(f"⏰ Exited recording hours (current time: {current_time.strftime('%H:%M:%S')}) - stopping camera...")
                        self._stop_camera()
                        camera_initialized = False
                    
                    # Sleep for a minute before checking time again
                    next_check = 60 - current_time.second  # Sleep until next minute
                    if self.verbose:
                        flush_print(f"😴 Outside recording hours (current time: {current_time.strftime('%H:%M:%S')}). Sleeping for {next_check}s...")
                    time.sleep(next_check)
        
        except Exception as e:
             if not self.stop_event.is_set():
                flush_print(f"❌ Fatal error in recording loop: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
        finally:
             if camera_initialized:
                 self._stop_camera()
             flush_print("🎥 Recording loop has finished.")

    def stop(self):
        """Signals all internal loops to stop and cleans up resources."""
        flush_print("🔄 Stopping recorder component...")
        self.stop_event.set()
        
        if self.frame_grabber_thread and self.frame_grabber_thread.is_alive():
            flush_print("⏳ Waiting for frame grabber thread to stop...")
            self.frame_grabber_thread.join(timeout=2)
            
        if self.picam2:
            try:
                self.picam2.stop()
                flush_print("📸 Camera stopped")
            except Exception as e:
                flush_print(f"⚠️ Warning: Error stopping camera: {e}")
        
        flush_print("✅ Recorder component stopped.") 