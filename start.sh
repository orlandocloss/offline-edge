#!/bin/bash

# Start the offline edge pipeline with maximum SSH responsiveness
# This ensures immediate output and responsive Ctrl+C, especially for hotspot connections

echo "$(date '+%Y-%m-%d %H:%M:%S') 🚀 Starting Offline Edge Pipeline with maximum SSH responsiveness..."
echo "$(date '+%Y-%m-%d %H:%M:%S') 📡 SSH-optimized for real-time output"
echo "$(date '+%Y-%m-%d %H:%M:%S') 🛑 Press Ctrl+C to stop gracefully"
echo ""

# Check if unbuffer is available (part of expect package)
if ! command -v unbuffer &> /dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') ⚠️ unbuffer not available - install 'expect' package when online for best hotspot performance"
    echo "$(date '+%Y-%m-%d %H:%M:%S') 💡 Run when online: sudo apt-get update && sudo apt-get install -y expect tmux"
fi

# Detect network interface type for hotspot-specific optimizations
HOTSPOT_INTERFACE=$(ip route | grep default | grep -E "(wlan0|ap0)" | head -1 | awk '{print $5}')
if [ -n "$HOTSPOT_INTERFACE" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') 📶 Detected hotspot interface: $HOTSPOT_INTERFACE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') 🔧 Applying hotspot-specific optimizations..."
    
    # Configure TCP keepalive for hotspot connections
    echo 30 > /proc/sys/net/ipv4/tcp_keepalive_time 2>/dev/null || true
    echo 5 > /proc/sys/net/ipv4/tcp_keepalive_intvl 2>/dev/null || true
    echo 3 > /proc/sys/net/ipv4/tcp_keepalive_probes 2>/dev/null || true
    
    # Additional hotspot-specific settings
    export HOTSPOT_MODE=1
    echo "$(date '+%Y-%m-%d %H:%M:%S') ✅ Hotspot mode optimizations enabled"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') 🔗 Standard network connection detected"
    export HOTSPOT_MODE=0
fi

# Activate virtual environment
echo "$(date '+%Y-%m-%d %H:%M:%S') 🔄 Activating virtual environment..."
source ../sensing-garden/venv_hailo_rpi5/bin/activate

# Ensure we're in the right directory
cd /home/sg/offline-edge

echo "$(date '+%Y-%m-%d %H:%M:%S') 📁 Using USB drive for storage: /media/sg/92A9-FB17/"
echo "$(date '+%Y-%m-%d %H:%M:%S') ⚙️  Configuration: 30s videos, 10fps, 10% sanity videos"
echo "$(date '+%Y-%m-%d %H:%M:%S') 🕘 Current time: $(date '+%H:%M:%S') (Recording: 06:00-22:00)"
echo ""

# Force completely unbuffered output with hotspot-specific settings
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONFAULTHANDLER=1

# Additional hotspot-specific environment variables
if [ "$HOTSPOT_MODE" = "1" ]; then
    export PYTHONFLUSHSTDOUT=1
    export PYTHONFLUSHSTDERR=1
    echo "$(date '+%Y-%m-%d %H:%M:%S') 🌐 Hotspot mode: Enhanced output flushing enabled"
fi

# Use exec to replace shell process and eliminate any shell buffering
echo "$(date '+%Y-%m-%d %H:%M:%S') 🚀 Launching Python process with maximum output responsiveness..."

# Enhanced stdbuf settings for hotspot mode
if [ "$HOTSPOT_MODE" = "1" ]; then
    # Use unbuffer if available, otherwise use stdbuf only
    if command -v unbuffer &> /dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') 🔧 Using unbuffer for hotspot mode..."
        exec stdbuf -i0 -o0 -e0 unbuffer python3 -u -X dev run.py \
            --video-dir /media/sg/92A9-FB17/videos \
            --duration 30 \
            --sanity-video-percentage 10 \
            --device-id offline-edge \
            --fps 10 \
            --detections-dir /media/sg/92A9-FB17/detections \
            --sanity-videos-dir /media/sg/92A9-FB17/sanity-videos \
            --verbose 
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') 🔧 Using stdbuf-only mode for hotspot (install 'expect' package for better performance)"
        exec stdbuf -i0 -o0 -e0 python3 -u -X dev run.py \
            --video-dir /media/sg/92A9-FB17/videos \
            --duration 30 \
            --sanity-video-percentage 10 \
            --device-id offline-edge \
            --fps 10 \
            --detections-dir /media/sg/92A9-FB17/detections \
            --sanity-videos-dir /media/sg/92A9-FB17/sanity-videos \
            --verbose 
    fi
else
    exec stdbuf -i0 -o0 -e0 python3 -u -X dev run.py \
        --video-dir /media/sg/92A9-FB17/videos \
        --duration 30 \
        --sanity-video-percentage 10 \
        --device-id offline-edge \
        --fps 10 \
        --detections-dir /media/sg/92A9-FB17/detections \
        --sanity-videos-dir /media/sg/92A9-FB17/sanity-videos \
        --verbose 
fi 