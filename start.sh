#!/bin/bash

# Start the offline edge pipeline 
# Optimized for tmux usage - you can detach with Ctrl+b d

echo "🚀 Starting Offline Edge Pipeline..."
echo "🛑 Press Ctrl+C to stop gracefully"
echo "📋 In tmux: Ctrl+b d to detach, 'tmux attach-session -t pipeline' to reattach"
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source ../sensing-garden/venv_hailo_rpi5/bin/activate

# Ensure we're in the right directory
cd /home/sg/offline-edge

echo "📁 Using USB drive for storage: /media/sg/92A9-FB17/"
echo "⚙️  Configuration: 30s videos, 10fps, 10% sanity videos"
echo "🕘 Current time: $(date '+%H:%M:%S') (Recording: 06:00-22:00)"
echo ""

# Force completely unbuffered output
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

echo "🚀 Launching pipeline..."

# Run without exec so shell remains for tmux
stdbuf -i0 -o0 -e0 python3 -u run.py \
    --video-dir /media/sg/92A9-FB17/videos \
    --duration 30 \
    --sanity-video-percentage 10 \
    --device-id offline-edge \
    --fps 10 \
    --detections-dir /media/sg/92A9-FB17/detections \
    --sanity-videos-dir /media/sg/92A9-FB17/sanity-videos \
    --verbose 