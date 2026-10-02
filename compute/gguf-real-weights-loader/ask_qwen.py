#!/usr/bin/env python3
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

MODEL_PATH = "/home/fbetancourt/Gemini/models/qwen2.5-coder-1.5b-instruct-q4_0.gguf"
TOKENIZER_BIN = "/home/fbetancourt/Applications/llama.cpp/bin/llama-tokenize"
CONFIG_DIR = os.path.expanduser("~/.config/local_llm")
PROFILE_FILE = os.path.join(CONFIG_DIR, "profile.md")
FACTS_FILE = os.path.join(CONFIG_DIR, "facts.md")
LOCK_FILE = os.path.join(CONFIG_DIR, "facts.lock")
MANIFEST_FILE = os.path.join(CONFIG_DIR, "memory_manifest.json")

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

def tokenize(text: str):
    if not os.path.exists(TOKENIZER_BIN) or not os.path.exists(MODEL_PATH):
        raise RuntimeError("llama-tokenize or GGUF model not found.")
    cmd = [TOKENIZER_BIN, "-m", MODEL_PATH, "-p", text]
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
        return ""  # Incomplete or truncated line without newline
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
    return f"<|im_start|>system\n{combined}<|im_end|>\n"

def get_model_sha256():
    cache_file = os.path.join(CONFIG_DIR, "model_sha256.cache")
    if not os.path.exists(MODEL_PATH):
        return "not_found"
    st = os.stat(MODEL_PATH)
    cur_key = f"{st.st_size}_{st.st_mtime}"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                c_key, c_hash = f.read().strip().split(":", 1)
                if c_key == cur_key:
                    return c_hash
        except Exception:
            pass

    # Compute and cache
    h = hashlib.sha256()
    with open(MODEL_PATH, "rb") as f:
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
        "model_path": MODEL_PATH,
        "model_sha256": get_model_sha256(),
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
        tokens = tokenize(sys_text)
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

        # 1. Leer hechos existentes
        existing_facts = ""
        if os.path.exists(FACTS_FILE):
            with open(FACTS_FILE, "r", encoding="utf-8") as f:
                existing_facts = f.read()

        if not existing_facts.strip():
            new_facts_content = f"# Verified Facts & Learned Context\n- {fact.strip()}\n"
        else:
            new_facts_content = existing_facts.rstrip() + f"\n- {fact.strip()}\n"

        # 2. Construir prompt del sistema candidato
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

        # 3. Tokenizar y enviar a GPU (Fase 1: Prepare & Prefill)
        tokens = tokenize(staged_sys_text)
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

        # 4. Fase 2 (Commit): GPU confirmó OK -> reemplazar atómicamente facts.md con fsync
        tmp_file = FACTS_FILE + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(new_facts_content)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, FACTS_FILE)

        # Fsync directorio para garantizar persistencia física del journaling
        dir_fd = os.open(CONFIG_DIR, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

        # 5. Persistir Memory Manifest
        write_manifest(b_gen, b_hash, b_pos, prompt_hash=prompt_hash, last_fact=fact.strip())

        print(f"✅ Fase 2 (Commit): Memoria persistida atómicamente en {FACTS_FILE}")
        print(f"📄 Manifiesto actualizado: {MANIFEST_FILE} (Gen {b_gen}, Hash {b_hash})")
        print(f"[*] Servidor GPU: {resp}")
        return True
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

def main():
    parser = argparse.ArgumentParser(description="ask-qwen: Local GPU AI CLI on NVIDIA GT 750M via qwen_server")
    parser.add_argument("prompt", nargs="?", default="", help="Prompt to send to local Qwen LLM")
    parser.add_argument("-n", "--tokens", type=int, default=64, help="Maximum new tokens to generate (default: 64)")
    parser.add_argument("--remember", help="Store a persistent fact in local memory and update GPU KV cache")
    parser.add_argument("--status", action="store_true", help="Check daemon and VRAM health status")
    parser.add_argument("--sync-memory", action="store_true", help="Force prefill of profile.md and facts.md into GPU KV cache")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose progress logs")
    args = parser.parse_args()

    if args.remember:
        remember_fact(args.remember)
        return 0

    if args.status:
        s = connect_socket()
        if not s:
            print("❌ Daemon inactivo o socket no encontrado.", file=sys.stderr)
            return 1
        st = get_status(s)
        s.close()
        print(f"📊 Estado del Daemon Qwen (GT 750M Kepler OpenCL 3.0):\n  {st}")
        if os.path.exists(MANIFEST_FILE):
            try:
                with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                    man = json.load(f)
                print(f"📄 Manifiesto en Disco:")
                print(f"  Gen: {man.get('base_generation')} | Hash: {man.get('base_hash')} | Tokens: {man.get('base_pos')} | Estado: {man.get('memory_state')}")
                print(f"  Model SHA256: {man.get('model_sha256')}")
                print(f"  Tokenizer Hash: {man.get('tokenizer_hash')} | Template Hash: {man.get('chat_template_hash')}")
                print(f"  Actualizado: {man.get('updated_at')}")
            except Exception:
                pass
        return 0

    if args.sync_memory:
        ok = sync_base_memory(verbose=True)
        return 0 if ok else 1

    if not args.prompt:
        parser.print_help()
        return 1

    s = connect_socket()
    if not s:
        print("❌ Error: qwen_server no responde en el socket.", file=sys.stderr)
        print("   Inicia el servidor con: RUSTICL_ENABLE=nouveau ./qwen_server", file=sys.stderr)
        return 1

    # Check if base memory is loaded and matches disk manifest
    st = get_status(s)
    s.close()

    # Active verification: check if server loaded the same model weights as on disk
    m_srv_sha = re.search(r'model_sha256=([0-9a-fA-F]+)', st)
    if m_srv_sha:
        srv_sha = m_srv_sha.group(1).lower()
        cli_sha = get_model_sha256().lower()
        if srv_sha != cli_sha and srv_sha != "unknown":
            print("❌ Error crítico: El modelo GGUF en disco ha cambiado respecto al residente en VRAM.", file=sys.stderr)
            print(f"   Disco: {cli_sha}\n   VRAM:  {srv_sha}", file=sys.stderr)
            print("   Reinicia el servidor para cargar los nuevos pesos: RUSTICL_ENABLE=nouveau ./qwen_server", file=sys.stderr)
            return 1

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
            manifest_hash = str(man.get("base_hash", "")).lower()
            if not server_hash or server_hash != manifest_hash:
                need_rebuild = True
            elif man.get("prompt_content_hash") != cur_prompt_hash:
                need_rebuild = True
            elif man.get("model_sha256") != get_model_sha256():
                need_rebuild = True
            elif man.get("tokenizer_hash") != get_tokenizer_hash():
                need_rebuild = True
            elif man.get("chat_template_hash") != get_chat_template_hash():
                need_rebuild = True
        except Exception:
            need_rebuild = True

    if need_rebuild:
        print("[!] Memoria en VRAM no sincronizada o no inicializada. Reconstruyendo KV cache en GPU VRAM desde disco...", file=sys.stderr)
        ok = sync_base_memory(verbose=True)
        if not ok:
            print("❌ Error reconstruyendo memoria base en VRAM.", file=sys.stderr)
            return 1

    # Format user prompt in ChatML
    user_chatml = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
    q_tokens = tokenize(user_chatml)

    s = connect_socket()
    if not s:
        print("❌ Error reconectando al socket.", file=sys.stderr)
        return 1

    req = f"QUERY {args.tokens} {','.join(map(str, q_tokens))}\n"
    s.sendall(req.encode())

    # Stream output
    try:
        while True:
            chunk = s.recv(1024)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            if text == "ERR_CONTEXT_FULL\n":
                print("\n❌ Error: El contexto excede el límite máximo (T_MAX=4096).", file=sys.stderr)
                break
            elif text == "ERR_BUSY\n":
                print("\n⚠️  Servidor ocupado procesando otra consulta.", file=sys.stderr)
                break
            elif text == "ERR_NOT_READY\n":
                print("\n⏳ Servidor en proceso de reconstrucción (REBUILDING). Intenta en unos segundos.", file=sys.stderr)
                break
            sys.stdout.write(text)
            sys.stdout.flush()
        print()
    finally:
        s.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
