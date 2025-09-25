# vLLM Self-Hosted Deployment on Protected HPC Clusters

> **Challenge**: Deploy OpenAI-compatible vLLM service on compute nodes with network restrictions  
> **Solution**: Double-hop SSH tunneling (Reverse + Local) for stable access through firewalls

## Architecture

```
[Local Machine] ──ssh -L──> [Campus Bastion] <──ssh -R── [Compute Node]
                             (Jump Host)                   (vLLM Service)
```

## Key Features

- ✅ **Zero-exposure deployment**: vLLM binds only to localhost (127.0.0.1)
- ✅ **Firewall-friendly**: Works with restricted HPC environments  
- ✅ **Production-ready**: Automatic network discovery, structured logging, health checks
- ✅ **OpenAI Compatible**: Full `/v1/*` API compatibility

## Quick Start

### 1. Deploy vLLM on Compute Node

```bash
# Start vLLM with optimized settings for Quadro RTX 8000 (SM 7.5)
python serve_vllm_openai.py \
  --host 127.0.0.1 --port 8000 \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --tp 1 --max-len 8192 --gpu-mem 0.90

# Detach to background (preserving service)
Ctrl+Z && bg && disown -h %1
```

### 2. Establish Tunnels

```bash
# On Compute Node: Reverse tunnel to bastion
ssh -fN -R 8020:127.0.0.1:8000 user@bastion.example.edu

# On Local Machine: Forward from bastion
ssh -fN -L 18000:127.0.0.1:8020 user@bastion.example.edu
```

### 3. Access Service

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:18000/v1", api_key="EMPTY")
print(client.models.list())
```

## Engineering Highlights

### Network Auto-Discovery
- Automatic detection of reachable jump hosts via `nc` port probing
- Smart fallback to alternative routes when primary path fails

### Observability
- Structured logging to `enterprise.jsonl` for audit trails
- Startup diagnostics with LAN/Public IP discovery
- Ready-to-copy curl/SDK examples in console output

### Resilience
- SSH keepalive parameters prevent connection drops
- `ExitOnForwardFailure` ensures clean tunnel establishment
- Process management without tmux/screen dependencies

## Troubleshooting

| Issue | Check | Command |
|-------|-------|---------|
| Service health | Direct access on compute node | `curl http://127.0.0.1:8000/v1/models` |
| Tunnel status | Port listening on bastion | `ss -tlnp \| grep 8020` |
| Connection | Local endpoint | `curl http://127.0.0.1:18000/v1/models` |

## Security Considerations

- **Minimal attack surface**: No direct network exposure
- **Authentication ready**: Support for `--api-key` when needed
- **Audit compliance**: Full request/response logging capability

## Technical Decisions

1. **FP16 by default** for SM 7.5 GPUs - optimal performance/stability trade-off
2. **Double-hop over VPN** - Works in environments where VPN is restricted
3. **Reverse tunnel first** - Bypasses compute→login routing limitations

---

### Environment Variables Reference

```bash
export VLLM_HOST=127.0.0.1      # Always localhost for security
export VLLM_PORT=8000           # vLLM service port
export R_TUN_PORT=8020          # Bastion reverse tunnel port
export L_TUN_PORT=18000         # Local forward port
```

