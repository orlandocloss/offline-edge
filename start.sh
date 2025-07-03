#!/bin/bash

# Start the offline edge pipeline with unbuffered output for SSH
# This ensures immediate output and responsive Ctrl+C

echo "🚀 Starting Offline Edge Pipeline with unbuffered output..."
echo "📡 SSH-optimized for immediate responsiveness"
echo "🛑 Press Ctrl+C to stop gracefully"
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source ../sensing-garden/venv_hailo_rpi5/bin/activate

# Ensure we're in the right directory
cd /home/sg/offline-edge

echo "📁 Using USB drive for storage: /media/sg/92A9-FB17/"
echo "⚙️  Configuration: 30s videos, 10fps, 10% sanity videos"
echo ""

# Run with unbuffered output (-u flag)
# stdbuf ensures all output is immediately flushed
exec stdbuf -oL -eL python3 -u run.py \
    --video-dir /media/sg/92A9-FB17/videos \
    --duration 30 \
    --sanity-video-percentage 10 \
    --device-id offline-edge \
    --fps 10 \
    --detections-dir /media/sg/92A9-FB17/detections \
    --sanity-videos-dir /media/sg/92A9-FB17/sanity-videos \
    --verbose 