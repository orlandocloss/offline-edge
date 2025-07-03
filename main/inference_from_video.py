import cv2
import time
import os
import sys
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from models.object_detection_utils import ObjectDetectionUtils
from models.detection import run_inference
from models.insect_tracker import InsectTracker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoInferenceProcessor:
    def __init__(self, detections_dir="detections", sanity_videos_dir="sanity_videos",
                 model_path="weights/small-generic.hef", labels_path="data/labels.txt",
                 batch_size=1, confidence_threshold=0.35, device_id="video_processor"):
        
        # Model and processor configuration
        self.model_path = model_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.device_id = device_id
        
        # Local storage directories
        self.detections_dir = Path(detections_dir)
        self.sanity_videos_dir = Path(sanity_videos_dir)
        
        # Create directories if they don't exist
        self.detections_dir.mkdir(exist_ok=True, parents=True)
        self.sanity_videos_dir.mkdir(exist_ok=True, parents=True)
        
        # Download weights if they don't exist
        self._ensure_weights_exist()
        
        # Per-video frame count for logging/timestamps
        self.frame_count = 0
        self.detection_count = 0
        
        self.det_utils = ObjectDetectionUtils(labels_path)
        
    def _ensure_weights_exist(self):
        """Check if weight files exist and download them if they don't."""
        weights_info = [
            {
                'path': self.model_path,
                'url': 'https://github.com/aasehaa/sensing-garden/releases/download/weights/small-generic.hef'
            }
        ]
        
        # Create weights directory if it doesn't exist
        weights_dir = os.path.dirname(self.model_path)
        if weights_dir and not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
            print(f"Created directory: {weights_dir}")
        
        for weight_info in weights_info:
            weight_path = weight_info['path']
            weight_url = weight_info['url']
            
            if not os.path.exists(weight_path):
                print(f"Weight file not found: {weight_path}")
                print(f"Downloading from: {weight_url}")
                
                try:
                    import requests
                    response = requests.get(weight_url, stream=True)
                    response.raise_for_status()
                    
                    # Get file size for progress tracking
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0
                    
                    with open(weight_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                
                                # Show progress every 10MB
                                if total_size > 0 and downloaded_size % (10 * 1024 * 1024) == 0:
                                    progress = (downloaded_size / total_size) * 100
                                    print(f"Download progress: {progress:.1f}% ({downloaded_size // (1024*1024)}MB/{total_size // (1024*1024)}MB)")
                    
                    print(f"Successfully downloaded: {weight_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to download {weight_path}: {e}")
                    raise RuntimeError(f"Could not download required weight file: {weight_path}")
            else:
                print(f"Weight file found: {weight_path}")
    
    def convert_bbox_to_normalized(self, x, y, x2, y2, width, height):
        x_center = (x + x2) / 2.0 / width
        y_center = (y + y2) / 2.0 / height
        norm_width = (x2 - x) / width
        norm_height = (y2 - y) / height
        return [x_center, y_center, norm_width, norm_height]
    
    def save_detection_locally(self, frame, detection_data, timestamp, frame_time_seconds):
        """Save detection frame and data locally."""
        try:
            # Create timestamp-based filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            
            # Save frame as JPG
            frame_filename = f"{timestamp_str}.jpg"
            frame_path = self.detections_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            # Save detection data as TXT
            txt_filename = f"{timestamp_str}.txt"
            txt_path = self.detections_dir / txt_filename
            
            with open(txt_path, 'w') as f:
                f.write(f"timestamp: {timestamp}\n")
                f.write(f"frame_time_seconds: {frame_time_seconds:.3f}\n")
                f.write(f"device_id: {self.device_id}\n")
                f.write(f"track_id: {detection_data.get('track_id', 'None')}\n")
                f.write(f"confidence: {detection_data.get('confidence', 0.0):.3f}\n")
                
                # Save normalized bounding box
                bbox = detection_data.get('bbox', [])
                if bbox:
                    f.write(f"bbox_normalized: {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
                # Save pixel coordinates
                pixel_coords = detection_data.get('pixel_coords', [])
                if pixel_coords:
                    f.write(f"bbox_pixels: {pixel_coords[0]} {pixel_coords[1]} {pixel_coords[2]} {pixel_coords[3]}\n")
            
            self.detection_count += 1
            print(f"💾 Saved detection {self.detection_count}: {frame_filename} and {txt_filename}")
            sys.stdout.flush()

        except Exception as e:
            print(f"Error saving detection locally: {e}")
    
    def process_frame(self, frame, frame_time_seconds, tracker, global_frame_count, show_boxes=False):
        """Process a single frame from the video."""
        # Increment per-video frame counter for logging
        self.frame_count += 1
        
        # Convert BGR to RGB for inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        infer_results = run_inference(
            net=self.model_path,
            input=rgb_frame,
            batch_size=self.batch_size,
            labels=self.labels_path,
            save_stream_output=False
        )
        
        if show_boxes:
            print(f"Frame {self.frame_count} ({frame_time_seconds:.2f}s): Found {len(infer_results)} raw detections")
        
        # First pass: collect all valid detections for tracking
        valid_detections = []
        valid_detection_data = []
        
        if len(infer_results) > 0:
            height, width = frame.shape[:2]
            
            for detection in infer_results:
                if len(detection) != 5:
                    continue
                    
                y_min, x_min, y_max, x_max, confidence = detection
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Convert to pixel coordinates
                x, y = int(x_min * width), int(y_min * height)
                x2, y2 = int(x_max * width), int(y_max * height)
                
                # Clamp coordinates
                x, y, x2, y2 = max(0, x), max(0, y), min(width, x2), min(height, y2)
                
                if x2 <= x or y2 <= y:
                    continue
                
                # Store detection for tracking (x1, y1, x2, y2 format)
                valid_detections.append([x, y, x2, y2])
                valid_detection_data.append({
                    'detection': detection,
                    'x': x, 'y': y, 'x2': x2, 'y2': y2,
                    'confidence': confidence
                })
        
        # Update tracker with detections using the global frame count
        track_ids = tracker.update(valid_detections, global_frame_count)
        
        if show_boxes and len(valid_detections) > 0:
            print(f"Frame {self.frame_count}: {len(valid_detections)} detections → {len(tracker.current_tracks)} active tracks")
        
        # Process each detection with its track ID
        for i, det_data in enumerate(valid_detection_data):
            x, y, x2, y2 = det_data['x'], det_data['y'], det_data['x2'], det_data['y2']
            confidence = det_data['confidence']
            track_id = track_ids[i] if i < len(track_ids) else None
            
            # Draw bounding box and track ID if showing boxes
            if show_boxes:
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence:.2f}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if track_id is not None:
                    cv2.putText(frame, f"ID:{track_id}", (x, y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Prepare detection data for saving
            detection_data = {
                "confidence": confidence,
                "bbox": self.convert_bbox_to_normalized(x, y, x2, y2, width, height),
                "pixel_coords": [x, y, x2, y2],
                "track_id": track_id
            }
            
            timestamp = datetime.now().isoformat()
            self.save_detection_locally(frame, detection_data, timestamp, frame_time_seconds)
        
        return frame

def process_video(video_path, processor, tracker, start_frame_count, show_video=False, output_video_path=None):
    """Process an MP4 video file frame by frame."""
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Processing video: {video_path}")
    print(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
    
    # Set up output video writer if requested
    out = None
    if output_video_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_video_path}")
    
    frame_number = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_time_seconds = frame_number / fps if fps > 0 else 0
            
            # Process the frame, passing the shared tracker and the correct global frame count
            global_frame_for_this_video = start_frame_count + frame_number
            processed_frame = processor.process_frame(frame, frame_time_seconds, tracker, global_frame_for_this_video, show_boxes=show_video)
            
            # Write to output video if requested
            if out:
                out.write(processed_frame)
            
            # Show video if requested
            if show_video:
                cv2.imshow('Video Inference', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User requested quit")
                    break
            
            frame_number += 1
            
            # Progress update every 100 frames
            if frame_number % 100 == 0:
                progress = (frame_number / total_frames) * 100
                elapsed = time.time() - start_time
                estimated_total = (elapsed / frame_number) * total_frames if frame_number > 0 else 0
                remaining = estimated_total - elapsed
                print(f"Progress: {frame_number}/{total_frames} ({progress:.1f}%) - "
                      f"ETA: {remaining:.1f}s")
    
    finally:
        cap.release()
        if out:
            out.release()
        if show_video:
            cv2.destroyAllWindows()
    
    processing_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Processed {frame_number} frames in {processing_time:.2f}s")
    print(f"Average processing speed: {frame_number/processing_time:.2f} FPS")
    print(f"Total detections saved: {processor.detection_count}")
    return frame_number 