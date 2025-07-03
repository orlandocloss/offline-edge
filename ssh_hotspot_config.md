# SSH Hotspot Configuration Guide

## ⚠️ IMPORTANT: Install Required Packages First

**Before using hotspot mode, install required packages when online:**

```bash
sudo apt-get update
sudo apt-get install -y expect tmux htop iotop
```

See `OFFLINE_SETUP.md` for complete setup instructions.

## For Better SSH Responsiveness with Pi Hotspot Mode

### SSH Client Configuration

Add these settings to your SSH client configuration (`~/.ssh/config` on Linux/Mac):

```ssh
# Raspberry Pi Hotspot Optimizations
Host pi-hotspot
    HostName <your-pi-ip>
    User sg
    
    # Keep connections alive
    ServerAliveInterval 10
    ServerAliveCountMax 3
    
    # Reduce timeouts
    ConnectTimeout 10
    
    # Optimize for hotspot latency
    TCPKeepAlive yes
    
    # Disable problematic features that can cause delays
    GSSAPIAuthentication no
    UseDNS no
    
    # Force immediate output
    RequestTTY force
    
    # Use compression for better performance over wireless
    Compression yes
    CompressionLevel 6
```

### SSH Command Line Options

For one-time connections, use these flags:

```bash
ssh -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -o Compression=yes sg@<pi-ip>
```

### Terminal Settings

For best results, configure your terminal:

1. **Disable line buffering**:
   ```bash
   stty -icanon min 1 time 0
   ```

2. **Set terminal to raw mode** (if needed):
   ```bash
   stty raw -echo
   ```

### Running the Pipeline with Tmux (Recommended)

Use tmux for persistent sessions:

```bash
# Start tmux session
tmux new-session -d -s pipeline

# Attach to session
tmux attach-session -t pipeline

# Inside tmux, run the pipeline
./start.sh
```

**Tmux benefits:**
- Pipeline continues even if SSH disconnects
- Easy reconnection from any SSH session
- Multiple windows for monitoring

### Running the Pipeline Directly

When connected via hotspot, use the optimized start script:

```bash
./start.sh
```

The script will automatically detect hotspot mode and apply:
- Enhanced output flushing (3x more aggressive)
- TCP keepalive optimizations
- Network packet transmission delays
- Terminal attribute resets for immediate display

### Troubleshooting

If output is still delayed:

1. **Check if required packages are installed**:
   ```bash
   which unbuffer  # Should show: /usr/bin/unbuffer
   which tmux      # Should show: /usr/bin/tmux
   ```

2. **Check if you're actually using hotspot mode**:
   ```bash
   echo $HOTSPOT_MODE
   ```

3. **Verify the network interface**:
   ```bash
   ip route | grep default
   ```

4. **Test with manual flushing**:
   ```bash
   python3 -u -c "import sys; print('test', flush=True); sys.stdout.flush()"
   ```

### Expected Behavior

With these optimizations and packages installed:
- ✅ Real-time output (< 1 second delay)
- ✅ Immediate Ctrl+C response
- ✅ Continuous heartbeat messages
- ✅ Responsive status updates
- ✅ No output buffering delays
- ✅ Persistent tmux sessions 