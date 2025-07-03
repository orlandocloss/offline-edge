# Simple SSH & Tmux Usage Guide

## Running with Tmux (Essential)

**The key to being able to detach properly is using tmux:**

```bash
# Start tmux session
tmux new-session -d -s pipeline

# Attach to session
tmux attach-session -t pipeline

# Inside tmux, run the pipeline
./start.sh
```

## Tmux Controls

- **Detach**: `Ctrl+b` then `d` (pipeline keeps running)
- **Re-attach**: `tmux attach-session -t pipeline`
- **Kill session**: `tmux kill-session -t pipeline`

## Benefits

✅ **Persistent sessions** - Pipeline continues even if SSH disconnects  
✅ **Easy reconnection** - Attach from any SSH session  
✅ **Proper signal handling** - Ctrl+b d always works  

## SSH Client Optimization (Optional)

If you want better SSH performance, add to `~/.ssh/config`:

```ssh
Host pi
    HostName <your-pi-ip>
    User sg
    ServerAliveInterval 30
    TCPKeepAlive yes
```

## Expected Behavior

With tmux:
- ✅ `Ctrl+b d` detaches properly
- ✅ Pipeline continues running after detach
- ✅ Can reconnect anytime with `tmux attach-session -t pipeline`
- ✅ Multiple SSH sessions can monitor the same pipeline 