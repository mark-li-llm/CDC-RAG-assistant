#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launch vLLM OpenAI-compatible server (/v1/chat/completions) with enterprise-grade observability enhancements:
- Structured JSON logging (key operational metrics)
- Raw vLLM stdout/stderr file persistence
- Parse and aggregate memory/concurrency/graph capture core information
- Health check and readiness detection (/v1/models)
- Automatic dtype compatibility: Compute Capability < 8.0 (Volta/Turing) forces FP16
- Optional API Key enforcement; optional disable model's built-in generation config (equivalent to --generation-config vllm)

Example (single Turing card, FP16, 4096 context):
  export HF_TOKEN=your_hf_token
  export CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=0 python serve_vllm_openai.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --tp 1 --port 8000 --host 0.0.0.0 \
  --max-len 4096 --gpu-mem 0.85 \
  --log-dir ./logs --retention-days 7 \
  --disable-hf-generation-config

After startup, you can use curl (with stream:true) for streaming tests.
"""

import os
import sys
import re
import json
import time
import atexit
import signal
import socket
import shlex
import argparse
import subprocess
import threading
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import logging
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import platform  # Platform and runtime environment information

# -----------------------
# Logging and utilities
# -----------------------

def ensure_dir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        # Allow logger.info(json=...) to pass structured fields
        if hasattr(record, "json_payload") and isinstance(record.json_payload, dict):
            payload.update(record.json_payload)
        return json.dumps(payload, ensure_ascii=False)

def get_logger(log_dir: str, name: str = "enterprise"):
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(JsonFormatter())
    logger.addHandler(ch)

    # File (daily rotation, retention controlled by backupCount)
    fh = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "enterprise.jsonl"),
        when="midnight",
        backupCount=7,
        encoding="utf-8",
        utc=True,
    )
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)
    return logger

def log_json(logger, msg: str, **fields):
    logger.info(msg, extra={"json_payload": fields})

# -----------------------
# Network information collection and self-reporting
# -----------------------

def get_local_ips():
    """Enumerate local available IPv4 addresses without relying on third-party libraries (filter 127.x)"""
    ips = set()
    # Get default outbound interface IP via "UDP no-send" trick
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
    except Exception:
        pass
    finally:
        try:
            s.close()
        except Exception:
            pass
    # Hostname resolution
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None)
        for fam, *_rest, sa in infos:
            if fam == socket.AF_INET:
                ip = sa[0]
                if not ip.startswith("127."):
                    ips.add(ip)
    except Exception:
        pass
    # Optional: psutil for more comprehensive results (if available in environment)
    try:
        import psutil  # noqa: F401
        for nic, addrs in psutil.net_if_addrs().items():  # type: ignore
            for a in addrs:
                if getattr(a, "family", None) == socket.AF_INET:
                    ip = a.address
                    if ip and not ip.startswith("127."):
                        ips.add(ip)
    except Exception:
        pass
    return sorted(ips)

def get_public_ip(timeout=2):
    """Attempt to detect public outbound IP; returns None if no internet or blocked"""
    for url in ("https://api.ipify.org", "https://ifconfig.me/ip"):
        try:
            with urlopen(Request(url, headers={"User-Agent": "vllm-self-report"}), timeout=timeout) as r:
                txt = r.read().decode().strip()
                if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", txt):
                    return txt
        except Exception:
            continue
    return None

def print_and_log_connect_info(args, ent_logger, agg, served_model_name=None):
    local_ips = get_local_ips()
    public_ip = get_public_ip()
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    py_info = f"{sys.version.split()[0]}"
    container_runtime = None
    try:
        with open("/proc/1/cgroup", "rt") as f:
            c = f.read()
            if "docker" in c:
                container_runtime = "docker"
            elif "kubepods" in c:
                container_runtime = "kubernetes"
    except Exception:
        pass

    if args.host == "0.0.0.0":
        urls = [f"http://{ip}:{args.port}" for ip in (local_ips or ["<LAN_IP>"])]
    else:
        urls = [f"http://{args.host}:{args.port}"]

    # Structured logging
    log_json(
        ent_logger,
        "server.network",
        host=args.host,
        port=args.port,
        hostname=hostname,
        fqdn=fqdn,
        os=os_info,
        python=py_info,
        container_runtime=container_runtime,
        local_ips=local_ips,
        public_ip=public_ip,
        reachable_urls=urls,
    )

    # Write to aggregated metrics (will be persisted by startup_summary_*)
    agg.setdefault("network", {})
    agg["network"].update({
        "host": args.host, "port": args.port,
        "hostname": hostname, "fqdn": fqdn,
        "os": os_info, "python": py_info,
        "container_runtime": container_runtime,
        "local_ips": local_ips, "public_ip": public_ip,
        "urls": urls
    })

    # Terminal-friendly output (easy to copy)
    served = served_model_name or args.served_model_name or args.model
    ek = args.api_key or "<YOUR_API_KEY>"
    base_for_snippet = (urls[0] if urls else f"http://{args.host}:{args.port}")

    print("\n========== CONNECT FROM OTHER MACHINES ==========")
    print(f" Bind Host: {args.host}    Port: {args.port}")
    print(f" Hostname : {hostname}    FQDN: {fqdn}")
    print(f" OS/Python: {os_info} / {py_info}")
    if container_runtime:
        print(f" Runtime  : {container_runtime}")
    if local_ips:
        print(" LAN IPs  : " + ", ".join(local_ips))
    if public_ip:
        print(f" Public IP: {public_ip}  (For cross-internet access, configure port forwarding/security groups)")
    print(" URLs     :")
    for u in urls:
        print(f"  - {u}")

    print("\n# Quick connectivity check:")
    print(f"curl -s {base_for_snippet}/v1/models")

    print("\n# OpenAI style (curl, streaming):")
    chat_payload = json.dumps({
        "model": served,
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}]
    }, ensure_ascii=False)
    print(
        f"curl -N {base_for_snippet}/v1/chat/completions "
        f'-H "Content-Type: application/json" '
        f'-H "Authorization: Bearer {ek}" '
        f"-d '{chat_payload}'"
    )

    print("\n# Python client (openai>=1.0):")
    print(
        "from openai import OpenAI\n"
        f'client = OpenAI(base_url="{base_for_snippet}/v1", api_key="{ek}")\n'
        "print(client.models.list())\n"
        "resp = client.chat.completions.create(model=\"" + served + "\", messages=[{\"role\":\"user\",\"content\":\"hi\"}], stream=True)\n"
        "for chunk in resp: print(chunk)"
    )

    print("\n# Node (official openai):")
    print(
        "import OpenAI from 'openai';\n"
        f"const client = new OpenAI({{ baseURL: '{base_for_snippet}/v1', apiKey: '{ek}' }});\n"
        "const models = await client.models.list();\n"
        "console.log(models.data?.map(m=>m.id));"
    )
    print("\n(Tip) To access from other machines, ensure: firewall/security groups allow the port, service started with --host 0.0.0.0, configure port mapping or reverse proxy if needed.")
    print("=================================================\n")

# -----------------------
# Parse vLLM key logs
# -----------------------

PATTERNS = {
    "mem_profile_time": re.compile(r"Memory profiling takes ([\d\.]+) seconds"),
    "usable_mem": re.compile(
        r"current vLLM instance can use total_gpu_memory \(([\d\.]+)GiB\) x gpu_memory_utilization \(([\d\.]+)\) = ([\d\.]+)GiB"
    ),
    "mem_breakdown": re.compile(
        r"model weights take ([\d\.]+)GiB; non_torch_memory takes ([\d\.]+)GiB; PyTorch activation peak memory takes ([\d\.]+)GiB; the rest of the memory reserved for KV Cache is ([\d\.]+)GiB"
    ),
    "blocks": re.compile(r"# cuda blocks: (\d+), # CPU blocks: (\d+)"),
    "max_conc": re.compile(r"Maximum concurrency for (\d+)\s+tokens per request:\s+([\d\.]+)x"),
    "cudagraph_backend": re.compile(r"Using (XFormers|FlashAttention-2) backend\."),
    "graph_finish": re.compile(r"Graph capturing finished in (\d+)\s+secs, took ([\d\.]+)\s+GiB"),
    "engine_init": re.compile(r"init engine.*took ([\d\.]+) seconds"),
    "sampling_override": re.compile(r"Default sampling parameters have been overridden"),
    "engine_pid": re.compile(r"Started engine process with PID (\d+)"),
    "platform": re.compile(r"Automatically detected platform (\w+)"),
}

def parse_and_collect(line: str, agg: dict, logger):
    # Match key information and output as structured logs, accumulate to agg
    m = PATTERNS["mem_profile_time"].search(line)
    if m:
        agg["mem_profiling_seconds"] = float(m.group(1))
        log_json(logger, "vllm.mem_profiling", seconds=agg["mem_profiling_seconds"])

    m = PATTERNS["usable_mem"].search(line)
    if m:
        total, util, usable = map(float, m.groups())
        agg["gpu_total_gib"] = total
        agg["gpu_mem_util"] = util
        agg["gpu_usable_gib"] = usable
        log_json(logger, "vllm.mem_usable",
                 gpu_total_gib=total, gpu_mem_utilization=util, gpu_usable_gib=usable)

    m = PATTERNS["mem_breakdown"].search(line)
    if m:
        w, non_torch, act_peak, kv = map(float, m.groups())
        agg.update({
            "weights_gib": w,
            "non_torch_gib": non_torch,
            "activation_peak_gib": act_peak,
            "kv_cache_gib": kv,
        })
        log_json(logger, "vllm.mem_breakdown",
                 weights_gib=w, non_torch_gib=non_torch,
                 activation_peak_gib=act_peak, kv_cache_gib=kv)

    m = PATTERNS["blocks"].search(line)
    if m:
        cuda_blocks, cpu_blocks = map(int, m.groups())
        agg["cuda_blocks"] = cuda_blocks
        agg["cpu_blocks"] = cpu_blocks
        log_json(logger, "vllm.blocks", cuda_blocks=cuda_blocks, cpu_blocks=cpu_blocks)

    m = PATTERNS["max_conc"].search(line)
    if m:
        tokens, conc = m.groups()
        agg["max_concurrency_tokens"] = int(tokens)
        agg["max_concurrency_x"] = float(conc)
        log_json(logger, "vllm.max_concurrency",
                 tokens_per_request=int(tokens), max_concurrency_x=float(conc))

    m = PATTERNS["cudagraph_backend"].search(line)
    if m:
        backend = m.group(1).lower()
        agg["decode_backend"] = backend
        log_json(logger, "vllm.decode_backend", backend=backend)

    m = PATTERNS["graph_finish"].search(line)
    if m:
        secs, gib = m.groups()
        agg["graph_capture_seconds"] = int(secs)
        agg["graph_capture_gib"] = float(gib)
        log_json(logger, "vllm.graph_capture",
                 seconds=int(secs), memory_gib=float(gib))

    m = PATTERNS["engine_init"].search(line)
    if m:
        agg["engine_init_seconds"] = float(m.group(1))
        log_json(logger, "vllm.engine_init", seconds=agg["engine_init_seconds"])

    m = PATTERNS["sampling_override"].search(line)
    if m:
        agg["sampling_overridden"] = True
        log_json(logger, "vllm.sampling_overridden", hint="use --generation-config vllm to avoid this")

    m = PATTERNS["engine_pid"].search(line)
    if m:
        agg["engine_pid"] = int(m.group(1))
        log_json(logger, "vllm.engine_pid", pid=agg["engine_pid"])

    m = PATTERNS["platform"].search(line)
    if m:
        agg["platform"] = m.group(1)
        log_json(logger, "vllm.platform", platform=agg["platform"])

# -----------------------
# Health check
# -----------------------

def wait_until_ready(host: str, port: int, timeout: int, logger) -> float:
    url = f"http://{host}:{port}/v1/models"
    start = time.monotonic()
    while True:
        try:
            req = Request(url, headers={"Authorization": "Bearer READY-CHECK"})
            with urlopen(req, timeout=2) as r:
                if r.status == 200:
                    ready_s = time.monotonic() - start
                    log_json(logger, "server.ready", url=url, seconds=round(ready_s, 3))
                    return ready_s
        except (URLError, HTTPError, socket.error):
            pass
        if time.monotonic() - start > timeout:
            log_json(logger, "server.ready_timeout", url=url, timeout_seconds=timeout)
            return -1.0
        time.sleep(0.5)

# -----------------------
# Main process
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Launch vLLM OpenAI-compatible server (enterprise logging)")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="HF model name or local path")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size (single card: 1)")
    parser.add_argument("--port", type=int, default=8000, help="service port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="bind address")
    parser.add_argument("--max-len", type=int, default=8192, help="max model len (context length)")
    parser.add_argument("--dtype", type=str, default="auto", help="auto/half/float16/bfloat16/bf16/float32")
    parser.add_argument("--gpu-mem", type=float, default=0.90, help="GPU memory utilization estimate")
    parser.add_argument("--download-dir", type=str, default=None, help="weights/tokenizer cache directory (optional)")
    parser.add_argument("--uvicorn-log-level", type=str, default="info", help="Uvicorn log level: debug/info/warning/error/critical/trace")
    parser.add_argument("--api-key", type=str, default=None, help="(optional) enforce Authorization: Bearer <key> validation")
    parser.add_argument("--served-model-name", type=str, default=None, help="(optional) exposed model name")
    parser.add_argument("--health-timeout", type=int, default=120, help="readiness check timeout (seconds)")
    parser.add_argument("--log-dir", type=str, default="./logs", help="enterprise log directory (JSON and raw stdout)")
    parser.add_argument("--retention-days", type=int, default=7, help="JSON log retention days")
    parser.add_argument("--disable-hf-generation-config", action="store_true",
                        help="use vLLM default sampling (equivalent to --generation-config vllm), avoid model's HF gen config override")
    parser.add_argument("--enforce-eager", action="store_true", help="force eager mode, avoid CUDA Graph capture")
    args = parser.parse_args()

    # Log initialization
    ensure_dir(args.log_dir)
    ent_logger = get_logger(args.log_dir)
    # Adjust log retention days
    for h in ent_logger.handlers:
        if isinstance(h, TimedRotatingFileHandler):
            h.backupCount = args.retention_days

    # Basic environment recording
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(all visible)")
    log_json(ent_logger, "server.start",
             model=args.model, tp=args.tp, max_len=args.max_len, dtype=args.dtype,
             gpu_memory_utilization=args.gpu_mem,
             port=args.port, host=args.host,
             download_dir=args.download_dir,
             api_key_required=bool(args.api_key),
             disable_hf_generation_config=args.disable_hf_generation_config,
             enforce_eager=args.enforce_eager,
             hf_token_present=bool(hf_token),
             cuda_visible_devices=cuda_vis)

    # GPU/CC auto compatibility
    sm_major = sm_minor = None
    gpu_name = None
    total_mem_gb = None
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            sm_major, sm_minor = torch.cuda.get_device_capability(0)
            gpu_name = props.name
            total_mem_gb = round(props.total_memory / (1024**3), 2)
    except Exception as e:
        log_json(ent_logger, "gpu.inspect_failed", error=str(e))

    # < SM 8.0 force FP16
    if sm_major is not None and sm_major < 8:
        if args.dtype.lower() in ("auto", "bfloat16", "bf16"):
            log_json(ent_logger, "dtype.autofix",
                     reason="cc<8.0 not support bf16", from_dtype=args.dtype, to_dtype="half")
            args.dtype = "half"

    log_json(ent_logger, "gpu.info",
             gpu_name=gpu_name, compute_capability=f"{sm_major}.{sm_minor}" if sm_major is not None else None,
             total_mem_gb=total_mem_gb)

    # Assemble vLLM startup command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--tensor-parallel-size", str(args.tp),
        "--max-model-len", str(args.max_len),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_mem),
        "--trust-remote-code",
        "--port", str(args.port),
        "--host", args.host,
        "--uvicorn-log-level", args.uvicorn_log_level,
    ]
    if args.download_dir:
        cmd += ["--download-dir", args.download_dir]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    if args.served_model_name:
        cmd += ["--served-model-name", args.served_model_name]
    if args.enforce_eager:
        cmd += ["--enforce-eager"]
    if args.disable_hf_generation_config:
        # Avoid "Default sampling parameters have been overridden ..." warning
        cmd += ["--generation-config", "vllm"]

    # Raw stdout/stderr file persistence
    raw_log_path = os.path.join(args.log_dir, f"vllm_stdout_{args.port}.log")
    raw_fp = open(raw_log_path, "a", buffering=1, encoding="utf-8")

    log_json(ent_logger, "vllm.launch", command=cmd, raw_log=raw_log_path)
    print("[LAUNCH] ", shlex.join(cmd))

    # Start subprocess
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Graceful exit: forward signals
    def _forward(sig, frame):
        log_json(ent_logger, "signal.forward", signal=sig)
        try:
            proc.terminate()
        except Exception:
            pass
    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)
    atexit.register(lambda: proc.poll() is None and proc.terminate())

    # Aggregated metrics
    agg = {}

    # Async read subprocess logs
    def _reader():
        for line in proc.stdout:
            # Print stdout as-is
            sys.stdout.write(line)
            # Raw log persistence
            raw_fp.write(line)
            # Parse key metrics
            try:
                parse_and_collect(line, agg, ent_logger)
            except Exception as e:
                log_json(ent_logger, "parse.error", error=str(e), line=line.strip())

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    # Readiness check
    ready_seconds = wait_until_ready(args.host, args.port, args.health_timeout, ent_logger)
    agg["ready_seconds"] = ready_seconds

    # Network self-reporting (print & log after ready)
    print_and_log_connect_info(args, ent_logger, agg, served_model_name=args.served_model_name)

    # Output a startup summary (JSON)
    summary_path = os.path.join(args.log_dir, f"startup_summary_{args.port}.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump({
                "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                "model": args.model,
                "served_model_name": args.served_model_name,
                "port": args.port,
                "host": args.host,
                "dtype": args.dtype,
                "max_len": args.max_len,
                "tp": args.tp,
                "gpu_memory_utilization": args.gpu_mem,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(all visible)"),
                "gpu": {
                    "name": gpu_name,
                    "compute_capability": f"{sm_major}.{sm_minor}" if sm_major is not None else None,
                    "total_mem_gb": total_mem_gb,
                },
                "metrics": agg,
            }, sf, ensure_ascii=False, indent=2)
        log_json(ent_logger, "startup.summary_written", path=summary_path)
    except Exception as e:
        log_json(ent_logger, "startup.summary_failed", error=str(e))

    # Wait for subprocess to end
    rc = proc.wait()
    raw_fp.close()
    log_json(ent_logger, "server.exit", returncode=rc)
    sys.exit(rc)

if __name__ == "__main__":
    main()
