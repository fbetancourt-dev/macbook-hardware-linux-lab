#!/usr/bin/env python3
"""
ask-qwen: Unified CLI for Local Qwen AI on Mid-2014 MacBook Pro
Supports:
  - Hardware/Silicon target: GPU (NVIDIA GT 750M Kepler OpenCL), AVX (CPU 8T SIMD Fast via Ollama), CPU (Standard llama.cpp)
  - Models: 1.5B (qwen2.5-coder-1.5b), 0.5B (qwen2.5-0.5b), 3B, 7B
  - Persistent Memory: profile.md, facts.md journaling into GPU KV cache
  - Real-time streaming with live telemetry metrics
  - JSON output mode for programmatic/agent pipelines
  - Stdin piping / prompt files
"""

import os
import sys
import time
import json
import hashlib
import fcntl
import socket
import argparse
import subprocess
import re
import urllib.request
import urllib.error

# File paths
MODEL_PATH_15B = "/home/fbetancourt/Gemini/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf"
MODEL_PATH_05B = "/home/fbetancourt/Gemini/models/qwen2.5-0.5b-instruct-q4_0.gguf"

TOKENIZER_BIN = "/home/fbetancourt/Applications/llama.cpp/bin/llama-tokenize"
LLAMA_CLI_BIN = "/home/fbetancourt/Applications/llama.cpp/bin/llama-cli"

GENERATE_STREAM_15B = "/home/fbetancourt/Gemini/macbook-hardware-linux-lab/compute/gguf-real-weights-loader/generate_stream"
GENERATE_STREAM_05B = "/home/fbetancourt/Gemini/macbook-hardware-linux-lab/compute/gguf-real-weights-loader/generate_stream_05b"

CONFIG_DIR = os.path.expanduser("~/.config/local_llm")
PROFILE_FILE = os.path.join(CONFIG_DIR, "profile.md")
FACTS_FILE = os.path.join(CONFIG_DIR, "facts.md")
LOCK_FILE = os.path.join(CONFIG_DIR, "facts.lock")
MANIFEST_FILE = os.path.join(CONFIG_DIR, "memory_manifest.json")

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"

MODEL_MAP_OLLAMA = {
    "0.5b": "qwen2.5:0.5b",
    "0.5": "qwen2.5:0.5b",
    "1.5b": "qwen2.5-coder:1.5b",
    "1.5": "qwen2.5-coder:1.5b",
    "3b": "qwen2.5-coder:3b",
    "3": "qwen2.5-coder:3b",
    "7b": "qwen2.5-coder:7b",
    "7": "qwen2.5-coder:7b",
}

def resolve_model_path(model_key: str):
    k = model_key.lower().strip()
    if "0.5" in k:
        return MODEL_PATH_05B, "0.5b"
    return MODEL_PATH_15B, "1.5b"

def get_socket_path():
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg and os.path.isdir(xdg):
        sock = os.path.join(xdg, "qwen.sock")
        if os.path.exists(sock):
            return sock
    if os.path.exists("/tmp/qwen.sock"):
        return "/tmp/qwen.sock"
    if xdg and os.path.isdir(xdg):
        return os.path.join(xdg, "qwen.sock")
    return "/tmp/qwen.sock"

def connect_socket():
    sock_path = get_socket_path()
    if not os.path.exists(sock_path):
        return None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock_path)
        return s
    except Exception:
        return None

def tokenize(text: str, model_path: str = MODEL_PATH_15B):
    if not os.path.exists(TOKENIZER_BIN) or not os.path.exists(model_path):
        raise RuntimeError(f"llama-tokenize or GGUF model not found ({model_path}).")
    cmd = [TOKENIZER_BIN, "-m", model_path, "-p", text]
    res = subprocess.run(cmd, capture_output=True, text=True)
    tokens = []
    for line in res.stdout.splitlines():
        m = re.match(r'^\s*(\d+)\s*->', line)
        if m:
            tokens.append(int(m.group(1)))
    return tokens

def read_response_line(sock):
    buf = []
    while True:
        chunk = sock.recv(1024)
        if not chunk:
            break
        buf.append(chunk)
        if b'\n' in chunk:
            break
    raw = b''.join(buf)
    if not raw.endswith(b'\n'):
        return ""
    return raw.decode('utf-8', errors='replace').strip()

def get_status(sock):
    sock.sendall(b"STATUS\n")
    return read_response_line(sock)

def build_base_system_prompt():
    profile = ""
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            profile = f.read().strip()
    facts = ""
    if os.path.exists(FACTS_FILE):
        with open(FACTS_FILE, "r", encoding="utf-8") as f:
            facts = f.read().strip()
    
    parts = []
    if profile:
        parts.append(profile)
    if facts:
        parts.append(facts)
    combined = "\n\n".join(parts)
    return f"<|im_start|>system\n{combined}<|im_end|>\n" if combined else ""

def get_model_sha256(model_path: str = MODEL_PATH_15B):
    cache_file = os.path.join(CONFIG_DIR, f"{os.path.basename(model_path)}.sha256.cache")
    if not os.path.exists(model_path):
        return "not_found"
    st = os.stat(model_path)
    cur_key = f"{st.st_size}_{st.st_mtime}"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                c_key, c_hash = f.read().strip().split(":", 1)
                if c_key == cur_key:
                    return c_hash
        except Exception:
            pass

    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    val = h.hexdigest()
    try:
        with open(cache_file, "w") as f:
            f.write(f"{cur_key}:{val}\n")
    except Exception:
        pass
    return val

def get_tokenizer_hash():
    if not os.path.exists(TOKENIZER_BIN):
        return "not_found"
    h = hashlib.sha256()
    with open(TOKENIZER_BIN, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()[:16]

def get_chat_template_hash():
    tmpl = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return hashlib.sha256(tmpl.encode()).hexdigest()[:16]

def write_manifest(gen: int, b_hash: str, b_pos: int, prompt_hash: str = "", last_fact: str = ""):
    if not prompt_hash:
        prompt_content = build_base_system_prompt()
        prompt_hash = hashlib.sha256(prompt_content.encode("utf-8")).hexdigest()[:16]
    manifest = {
        "base_generation": gen,
        "base_hash": b_hash,
        "base_pos": b_pos,
        "prompt_content_hash": prompt_hash,
        "model_path": MODEL_PATH_15B,
        "model_sha256": get_model_sha256(MODEL_PATH_15B),
        "tokenizer_hash": get_tokenizer_hash(),
        "chat_template_hash": get_chat_template_hash(),
        "t_max": 4096,
        "memory_state": "READY",
        "last_fact_added": last_fact,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z")
    }
    man_tmp = MANIFEST_FILE + ".tmp"
    with open(man_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(man_tmp, MANIFEST_FILE)

    dir_fd = os.open(CONFIG_DIR, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

def sync_base_memory(verbose=False):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        s = connect_socket()
        if not s:
            print("❌ Error: qwen_server daemon no está activo.", file=sys.stderr)
            return False

        sys_text = build_base_system_prompt()
        prompt_hash = hashlib.sha256(sys_text.encode("utf-8")).hexdigest()[:16]
        tokens = tokenize(sys_text, MODEL_PATH_15B)
        if verbose:
            print(f"[*] Prefillando {len(tokens)} tokens de memoria base en GPU VRAM...", file=sys.stderr)

        req = f"SET_BASE {','.join(map(str, tokens))}\n"
        s.sendall(req.encode())
        resp = read_response_line(s)
        s.close()
        if verbose:
            print(f"[*] Servidor: {resp}", file=sys.stderr)

        m_pos = re.search(r'base_pos=(\d+)', resp)
        m_hash = re.search(r'base_hash=(0x[0-9a-fA-F]+)', resp)
        m_gen = re.search(r'base_generation=(\d+)', resp)
        if resp.startswith("OK ") and m_pos and m_hash and m_gen:
            b_pos = int(m_pos.group(1))
            b_hash = m_hash.group(1)
            b_gen = int(m_gen.group(1))
            write_manifest(b_gen, b_hash, b_pos, prompt_hash=prompt_hash)
            return True
        else:
            if verbose:
                print(f"❌ Respuesta inválida o incompleta del servidor: {resp}", file=sys.stderr)
            return False
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

def remember_fact(fact: str):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        existing_facts = ""
        if os.path.exists(FACTS_FILE):
            with open(FACTS_FILE, "r", encoding="utf-8") as f:
                existing_facts = f.read()

        if not existing_facts.strip():
            new_facts_content = f"# Verified Facts & Learned Context\n- {fact.strip()}\n"
        else:
            new_facts_content = existing_facts.rstrip() + f"\n- {fact.strip()}\n"

        profile = ""
        if os.path.exists(PROFILE_FILE):
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                profile = f.read().strip()

        parts = []
        if profile:
            parts.append(profile)
        parts.append(new_facts_content.strip())
        staged_sys_text = f"<|im_start|>system\n{'\n\n'.join(parts)}<|im_end|>\n"
        prompt_hash = hashlib.sha256(staged_sys_text.encode("utf-8")).hexdigest()[:16]

        tokens = tokenize(staged_sys_text, MODEL_PATH_15B)
        print(f"[*] Fase 1 (Prepare): Validando {len(tokens)} tokens candidatos en GPU VRAM...", file=sys.stderr)

        s = connect_socket()
        if not s:
            print("❌ Error: qwen_server daemon no responde. Memoria no modificada.", file=sys.stderr)
            return False

        req = f"SET_BASE {','.join(map(str, tokens))}\n"
        s.sendall(req.encode())
        resp = read_response_line(s)
        s.close()

        m_pos = re.search(r'base_pos=(\d+)', resp)
        m_hash = re.search(r'base_hash=(0x[0-9a-fA-F]+)', resp)
        m_gen = re.search(r'base_generation=(\d+)', resp)

        if not resp.startswith("OK ") or not (m_pos and m_hash and m_gen):
            print(f"❌ Error o respuesta incompleta en prefill de GPU: {resp}. Transacción abortada (disco intacto).", file=sys.stderr)
            return False

        b_pos = int(m_pos.group(1))
        b_hash = m_hash.group(1)
        b_gen = int(m_gen.group(1))

        tmp_file = FACTS_FILE + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(new_facts_content)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, FACTS_FILE)

        dir_fd = os.open(CONFIG_DIR, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

        write_manifest(b_gen, b_hash, b_pos, prompt_hash=prompt_hash, last_fact=fact.strip())

        print(f"✅ Fase 2 (Commit): Memoria persistida atómicamente en {FACTS_FILE}")
        print(f"📄 Manifiesto actualizado: {MANIFEST_FILE} (Gen {b_gen}, Hash {b_hash})")
        print(f"[*] Servidor GPU: {resp}")
        return True
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

# -------------------------------------------------------------
# EXECUTION BACKENDS
# -------------------------------------------------------------

def query_gpu_daemon(prompt: str, max_tokens: int, verbose: bool = False, output_json: bool = False):
    s = connect_socket()
    if not s:
        return None, "DAEMON_NOT_FOUND"

    st = get_status(s)
    s.close()

    m_val = re.search(r'model_sha256=(\S+)', st)
    raw_sha = m_val.group(1) if m_val else ""
    if not raw_sha or not re.fullmatch(r"[0-9a-fA-F]{64}", raw_sha):
        return None, f"INVALID_SERVER_SHA: {raw_sha}"

    if raw_sha.lower() != get_model_sha256(MODEL_PATH_15B).lower():
        return None, "MODEL_CHANGED_ON_DISK"

    # Check memory manifest sync
    need_rebuild = False
    m_server_hash = re.search(r'base_hash=(0x[0-9a-fA-F]+)', st)
    server_hash = m_server_hash.group(1).lower() if m_server_hash else ""

    cur_prompt = build_base_system_prompt()
    cur_prompt_hash = hashlib.sha256(cur_prompt.encode("utf-8")).hexdigest()[:16]

    if "base_pos=0" in st or "memory_state=READY" not in st:
        need_rebuild = True
    elif not os.path.exists(MANIFEST_FILE):
        need_rebuild = True
    else:
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                man = json.load(f)
            if not server_hash or server_hash != str(man.get("base_hash", "")).lower():
                need_rebuild = True
            elif man.get("prompt_content_hash") != cur_prompt_hash:
                need_rebuild = True
        except Exception:
            need_rebuild = True

    if need_rebuild:
        if verbose:
            print("[*] Reconstruyendo KV cache en GPU VRAM...", file=sys.stderr)
        if not sync_base_memory(verbose=verbose):
            return None, "FAILED_TO_REBUILD_MEMORY"

    user_chatml = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    q_tokens = tokenize(user_chatml, MODEL_PATH_15B)

    s = connect_socket()
    if not s:
        return None, "FAILED_TO_RECONNECT"

    req = f"QUERY {max_tokens} {','.join(map(str, q_tokens))}\n"
    s.sendall(req.encode())

    t_start = time.perf_counter()
    t_first_chunk = None
    chunks_received = 0
    full_text = []

    try:
        while True:
            chunk = s.recv(1024)
            if not chunk:
                break
            if t_first_chunk is None:
                t_first_chunk = time.perf_counter()
            chunks_received += 1
            text = chunk.decode("utf-8", errors="replace")
            if text in ("ERR_CONTEXT_FULL\n", "ERR_BUSY\n", "ERR_NOT_READY\n"):
                return None, text.strip()
            full_text.append(text)
            if not output_json:
                sys.stdout.write(text)
                sys.stdout.flush()
        if not output_json:
            print()
    finally:
        s.close()
        t_end = time.perf_counter()

    ttft = (t_first_chunk - t_start) if t_first_chunk else 0.0
    decode_time = (t_end - t_first_chunk) if t_first_chunk else (t_end - t_start)
    toks = max(1, chunks_received)
    rate = toks / decode_time if decode_time > 0.05 else 0.0
    ms_tok = (decode_time * 1000.0) / toks if toks > 0 and decode_time > 0 else 0.0

    metrics = {
        "device": "GPU (Kepler GT 750M)",
        "backend": "OpenCL 3.0 Rusticl (qwen_server)",
        "model": "qwen2.5-coder:1.5b",
        "ttft_s": ttft,
        "decode_s": decode_time,
        "total_s": t_end - t_start,
        "tokens": toks,
        "tokens_per_sec": rate,
        "ms_per_token": ms_tok,
        "vram_mb": 1110,
        "cpu_usage": "0% (Host idle)"
    }
    return "".join(full_text), metrics

def query_gpu_standalone(prompt: str, model_size: str, max_tokens: int, verbose: bool = False, output_json: bool = False, raw: bool = False):
    model_path, tag = resolve_model_path(model_size)
    gen_bin = GENERATE_STREAM_05B if tag == "0.5b" else GENERATE_STREAM_15B
    if not os.path.exists(gen_bin):
        return None, f"BINARY_NOT_FOUND: {gen_bin}"

    # Build ChatML prompt with system if exists
    sys_content = ""
    if not raw and os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            sys_content = f.read().strip()
    
    full_prompt = f"<|im_start|>system\n{sys_content}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" if sys_content else (prompt if raw else f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    tokens = tokenize(full_prompt, model_path)
    tok_str = ",".join(map(str, tokens))

    cmd = [gen_bin, "--tokens", tok_str, "-n", str(max_tokens), "--quiet", "--no-echo-prompt"]
    env = os.environ.copy()
    env["RUSTICL_ENABLE"] = "nouveau"

    t_start = time.perf_counter()
    bin_dir = os.path.dirname(gen_bin)
    p = subprocess.Popen(cmd, env=env, cwd=bin_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    t_first_chunk = None
    chunks_count = 0
    full_text = []

    for chunk in iter(lambda: p.stdout.read(1), ''):
        if not chunk:
            break
        if t_first_chunk is None:
            t_first_chunk = time.perf_counter()
        chunks_count += 1
        full_text.append(chunk)
        if not output_json:
            sys.stdout.write(chunk)
            sys.stdout.flush()

    p.stdout.close()
    p.wait()
    t_end = time.perf_counter()
    if not output_json:
        print()

    ttft = (t_first_chunk - t_start) if t_first_chunk else 0.0
    decode_time = (t_end - t_first_chunk) if t_first_chunk else (t_end - t_start)
    # Estimate tokens generated
    gen_str = "".join(full_text)
    toks = len(tokenize(gen_str, model_path)) if gen_str.strip() else max_tokens
    rate = toks / decode_time if decode_time > 0.05 else 0.0
    ms_tok = (decode_time * 1000.0) / toks if toks > 0 and decode_time > 0 else 0.0

    vram = 420 if tag == "0.5b" else 1110
    metrics = {
        "device": "GPU (Kepler GT 750M)",
        "backend": f"OpenCL 3.0 Rusticl ({os.path.basename(gen_bin)})",
        "model": f"qwen2.5:{tag}",
        "ttft_s": ttft,
        "decode_s": decode_time,
        "total_s": t_end - t_start,
        "tokens": toks,
        "tokens_per_sec": rate,
        "ms_per_token": ms_tok,
        "vram_mb": vram,
        "cpu_usage": "0% (Host idle)"
    }
    return gen_str, metrics

def query_avx_ollama(prompt: str, model_size: str, max_tokens: int, verbose: bool = False, output_json: bool = False, raw: bool = False):
    ollama_model = MODEL_MAP_OLLAMA.get(model_size.lower().strip(), f"qwen2.5-coder:{model_size}")
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "raw": raw,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "num_thread": 8
        }
    }
    # Add system prompt if available and not raw
    if not raw:
        sys_content = ""
        if os.path.exists(PROFILE_FILE):
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                sys_content = f.read().strip()
        if os.path.exists(FACTS_FILE):
            with open(FACTS_FILE, "r", encoding="utf-8") as f:
                f_txt = f.read().strip()
                if f_txt:
                    sys_content += "\n" + f_txt
        if sys_content:
            payload["system"] = sys_content

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_API_URL, data=req_data, headers={"Content-Type": "application/json"})

    t_start = time.perf_counter()
    t_first_chunk = None
    full_text = []
    eval_count = 0
    eval_duration_ns = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                if not line:
                    continue
                obj = json.loads(line.decode("utf-8"))
                part = obj.get("response", "")
                if part:
                    if t_first_chunk is None:
                        t_first_chunk = time.perf_counter()
                    full_text.append(part)
                    if not output_json:
                        sys.stdout.write(part)
                        sys.stdout.flush()
                if obj.get("done"):
                    eval_count = obj.get("eval_count", 0)
                    eval_duration_ns = obj.get("eval_duration", 0)
                    prompt_eval_duration_ns = obj.get("prompt_eval_duration", 0)
        if not output_json:
            print()
    except Exception as e:
        return None, f"OLLAMA_API_ERROR: {e}"

    t_end = time.perf_counter()
    if 'prompt_eval_duration_ns' in locals() and prompt_eval_duration_ns > 0:
        ttft = prompt_eval_duration_ns / 1e9
    else:
        ttft = (t_first_chunk - t_start) if t_first_chunk else 0.0

    if eval_duration_ns > 0:
        decode_time = eval_duration_ns / 1e9
    elif t_first_chunk:
        decode_time = t_end - t_first_chunk
    else:
        decode_time = t_end - t_start

    toks = eval_count if eval_count > 0 else max(1, len(full_text))
    rate = toks / decode_time if decode_time > 0.001 else 0.0
    ms_tok = (decode_time * 1000.0) / toks if toks > 0 and decode_time > 0 else 0.0

    metrics = {
        "device": "CPU (Intel Haswell i7-4870HQ)",
        "backend": "AVX2 + FMA3 SIMD (Ollama / libggml-cpu-haswell.so)",
        "model": ollama_model,
        "ttft_s": ttft,
        "decode_s": decode_time,
        "total_s": t_end - t_start,
        "tokens": toks,
        "tokens_per_sec": rate,
        "ms_per_token": ms_tok,
        "vram_mb": 0,
        "cpu_usage": "8 Threads AVX2"
    }
    return "".join(full_text), metrics

def query_cpu_standard(prompt: str, model_size: str, max_tokens: int, threads: int = 8, verbose: bool = False, output_json: bool = False):
    model_path, tag = resolve_model_path(model_size)
    if not os.path.exists(LLAMA_CLI_BIN):
        return None, f"BINARY_NOT_FOUND: {LLAMA_CLI_BIN}"
    if not os.path.exists(model_path):
        return None, f"MODEL_NOT_FOUND: {model_path}"

    cmd = [
        LLAMA_CLI_BIN,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "-t", str(threads),
        "--no-warmup",
        "--log-disable",
        "--single-turn",
        "--simple-io"
    ]

    t_start = time.perf_counter()
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    out, _ = p.communicate()
    t_end = time.perf_counter()

    # Clean banner and extract response
    lines = out.splitlines()
    clean_lines = []
    capture = False
    for line in lines:
        if line.startswith("> "):
            capture = True
            continue
        if "[ Prompt:" in line or "Exiting..." in line:
            break
        if capture and not line.startswith("Loading model"):
            clean_lines.append(line)

    result_text = "\n".join(clean_lines).strip()
    if not output_json:
        print(result_text)

    # Parse timings from output
    rate = 0.0
    m_rate = re.search(r'Generation:\s*([0-9.]+)\s*t/s', out)
    if m_rate:
        rate = float(m_rate.group(1))

    total_s = t_end - t_start
    toks = len(tokenize(result_text, model_path)) if result_text else max_tokens
    ms_tok = 1000.0 / rate if rate > 0.0 else 0.0
    decode_s = toks / rate if rate > 0.0 else total_s

    metrics = {
        "device": "CPU (Intel Haswell i7-4870HQ)",
        "backend": f"Standard llama-cli ({threads} Threads)",
        "model": f"qwen2.5:{tag}",
        "ttft_s": total_s - decode_s if total_s > decode_s else 0.2,
        "decode_s": decode_s,
        "total_s": total_s,
        "tokens": toks,
        "tokens_per_sec": rate,
        "ms_per_token": ms_tok,
        "vram_mb": 0,
        "cpu_usage": f"{threads} Threads Standard"
    }
    return result_text, metrics

# -------------------------------------------------------------
# MAIN CLI DISPATCHER
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ask-qwen: High-Performance Local AI CLI with Silicon & Acceleration Selection (GPU, AVX, CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  ask-qwen "Calcula un filtro pasa bajas RC a 10 kHz"
  ask-qwen -d gpu -m 1.5b "Escribe un driver I2C para ESP32"
  ask-qwen -d avx -m 1.5b "Explica el protocolo SPI en C"
  ask-qwen -d cpu -t 8 -m 1.5b "Compara velocidad con 8 hilos"
  ask-qwen -d cpu -m 0.5b "Hola Qwen"
  cat main.c | ask-qwen "Encuentra posibles memory leaks"
  ask-qwen --status
  ask-qwen --remember "El sensor de temperatura está en I2C dirección 0x48"
"""
    )

    parser.add_argument("prompt", nargs="?", default="", help="Prompt a enviar al modelo local Qwen")
    parser.add_argument("-d", "--device", choices=["gpu", "avx", "cpu", "cpu-fast"], default="gpu",
                        help="Silicon target / Motor de aceleración: 'gpu' (NVIDIA GT 750M Kepler), 'avx' (CPU 8T AVX2 Fast), 'cpu' (llama-cli estándar)")
    parser.add_argument("-m", "--model", default="1.5b",
                        help="Variante del modelo Qwen (ej: '1.5b', '0.5b', '3b', '7b'). Por defecto: 1.5b")
    parser.add_argument("-t", "--threads", type=int, default=8,
                        help="Número de hilos de CPU a utilizar para inferencia CPU (default: 8)")
    parser.add_argument("-n", "--tokens", type=int, default=64,
                        help="Máximo de tokens a generar (default: 64)")
    parser.add_argument("--json", action="store_true",
                        help="Emitir respuesta y telemetría en formato JSON (para automatización y agentes)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Mostrar logs de progreso y telemetría detallada en stderr")

    # Memory commands
    parser.add_argument("--remember", help="Registrar un hecho persistente en la memoria local (facts.md) y actualizar KV cache en VRAM")
    parser.add_argument("--status", action="store_true", help="Consultar estado de salud del daemon GPU y memoria en VRAM")
    parser.add_argument("--sync-memory", action="store_true", help="Forzar sincronización de profile.md y facts.md hacia el KV cache de la GPU")

    parser.add_argument("-f", "--file", help="Ruta a un archivo cuyo contenido se usará como prompt o contexto")
    parser.add_argument("--raw", action="store_true",
                        help="Completar texto puro directamente sin inyectar plantillas de chat o memoria del sistema (ideal para benchmarks)")

    args = parser.parse_args()

    # Memory administration
    if args.remember:
        ok = remember_fact(args.remember)
        return 0 if ok else 1

    if args.sync_memory:
        ok = sync_base_memory(verbose=True)
        return 0 if ok else 1

    if args.status:
        s = connect_socket()
        if not s:
            print("❌ Daemon GPU (qwen_server) inactivo o socket no encontrado en /tmp/qwen.sock.", file=sys.stderr)
            print("   Ollama AVX2 API:", file=sys.stderr)
            try:
                urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1)
                print("   ✅ Ollama local activo en puerto 11434 (AVX2 listo).", file=sys.stderr)
            except Exception:
                print("   ⚠️  Ollama inactivo.", file=sys.stderr)
            return 1
        st = get_status(s)
        s.close()
        print(f"📊 Estado del Daemon Qwen GPU (GT 750M Kepler OpenCL 3.0):\n  {st}")
        if os.path.exists(MANIFEST_FILE):
            try:
                with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                    man = json.load(f)
                print(f"📄 Manifiesto de Memoria en Disco:")
                print(f"  Gen: {man.get('base_generation')} | Hash: {man.get('base_hash')} | Tokens: {man.get('base_pos')} | Estado: {man.get('memory_state')}")
                print(f"  Model SHA256: {man.get('model_sha256')}")
                print(f"  Actualizado: {man.get('updated_at')}")
            except Exception:
                pass
        return 0

    # Resolve Prompt (CLI arg, File, or Stdin pipe)
    prompt_parts = []
    if args.file:
        if os.path.exists(args.file):
            with open(args.file, "r", encoding="utf-8") as f:
                prompt_parts.append(f.read().strip())
        else:
            print(f"❌ Error: El archivo especificado '{args.file}' no existe.", file=sys.stderr)
            return 1

    if not sys.stdin.isatty():
        try:
            stdin_data = sys.stdin.read().strip()
            if stdin_data:
                prompt_parts.append(stdin_data)
        except Exception:
            pass

    if args.prompt and args.prompt.strip():
        prompt_parts.append(args.prompt.strip())

    prompt_text = "\n\n".join(prompt_parts).strip()

    if not prompt_text:
        parser.print_help()
        return 1

    # Route execution based on device
    target_dev = args.device.lower()
    if target_dev == "cpu-fast":
        target_dev = "avx"

    model_key = args.model.lower().strip()
    resp_text = None
    metrics = None

    if target_dev == "gpu":
        # Check if 1.5B daemon is running
        if model_key in ("1.5", "1.5b", "default"):
            if not args.raw:
                resp_text, metrics = query_gpu_daemon(prompt_text, args.tokens, verbose=args.verbose, output_json=args.json)
            # If daemon not running or raw requested, fallback to standalone direct GPU generator
            if resp_text is None:
                if args.verbose and not args.raw:
                    print(f"[*] qwen_server inactivo ({metrics}). Usando generador GPU directo (generate_stream)...", file=sys.stderr)
                resp_text, metrics = query_gpu_standalone(prompt_text, "1.5b", args.tokens, verbose=args.verbose, output_json=args.json, raw=args.raw)
        else:
            # 0.5B standalone GPU
            resp_text, metrics = query_gpu_standalone(prompt_text, model_key, args.tokens, verbose=args.verbose, output_json=args.json, raw=args.raw)

    elif target_dev == "avx":
        resp_text, metrics = query_avx_ollama(prompt_text, model_key, args.tokens, verbose=args.verbose, output_json=args.json, raw=args.raw)
        # If Ollama fails, fallback to CPU standard
        if resp_text is None:
            if args.verbose:
                print(f"[*] Fallback: Ollama AVX no disponible ({metrics}). Ejecutando en CPU standard...", file=sys.stderr)
            resp_text, metrics = query_cpu_standard(prompt_text, model_key, args.tokens, threads=args.threads, verbose=args.verbose, output_json=args.json)

    elif target_dev == "cpu":
        resp_text, metrics = query_cpu_standard(prompt_text, model_key, args.tokens, threads=args.threads, verbose=args.verbose, output_json=args.json)

    if resp_text is None:
        print(f"❌ Error ejecutando inferencia en backend {target_dev}: {metrics}", file=sys.stderr)
        return 1

    # Output formatting
    if args.json:
        out_payload = {
            "prompt": prompt_text,
            "response": resp_text,
            "metrics": metrics
        }
        print(json.dumps(out_payload, indent=2, ensure_ascii=False))
    else:
        # Telemetry footer banner
        ttft_s = metrics.get("ttft_s", 0.0)
        decode_s = metrics.get("decode_s", 0.0)
        total_s = metrics.get("total_s", 0.0)
        toks = metrics.get("tokens", 0)
        rate = metrics.get("tokens_per_sec", 0.0)
        ms_tok = metrics.get("ms_per_token", 0.0)
        dev = metrics.get("device", target_dev)
        mod = metrics.get("model", model_key)

        p_str = f"{ttft_s*1000.0:.0f} ms" if ttft_s < 1.0 else f"{ttft_s:.2f} s"
        r_str = f"{decode_s*1000.0:.0f} ms" if decode_s < 1.0 else f"{decode_s:.2f} s"
        t_str = f"{total_s*1000.0:.0f} ms" if total_s < 1.0 else f"{total_s:.2f} s"

        print(f"\n[⏱️  {dev} | Modelo: {mod} | TTFT: {p_str} | Resp: {r_str} ({toks} tok @ {ms_tok:.0f} ms/tok, {rate:.2f} t/s) | Total: {t_str}]", file=sys.stderr)

    return 0

if __name__ == "__main__":
    sys.exit(main())
