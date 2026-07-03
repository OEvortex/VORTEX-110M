"""Minimal FastAPI server for sandbox operations."""
import hmac, os, subprocess, pathlib, signal, threading, re, tempfile
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import uvicorn

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07')

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)

def _truncate_output(output: str, max_chars: int = 25000, head_ratio: float = 0.25) -> str:
    if len(output) <= max_chars:
        return output
    # Write full output to temp file so LLM can read specific sections
    spill_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', prefix='bash_output_', dir='/tmp', delete=False) as f:
            f.write(output)
            spill_path = f.name
    except Exception:
        pass
    head_budget = int(max_chars * head_ratio)
    tail_budget = max_chars - head_budget
    head = output[:head_budget]
    tail = output[-tail_budget:]
    total = len(output)
    omitted = total - max_chars
    meta = f"\n\n... ({omitted:,} of {total:,} chars omitted, showing first {head_budget:,} + last {tail_budget:,}) ...\n"
    if spill_path:
        meta += f"Full output saved to {spill_path} — use the read tool with offset/limit to inspect specific sections.\n"
    return head + meta + tail

def _atomic_write(path: pathlib.Path, content: str):
    """Write atomically: temp file + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(tmp_path, str(path))
        tmp_path = None
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

app = FastAPI()

def _bearer_token(header: str) -> str:
    scheme, _, supplied = header.partition(" ")
    if scheme.lower() != "bearer" or not supplied:
        return ""
    return supplied

def _require_auth(request: Request) -> None:
    sandbox_token = os.environ.get("SANDBOX_API_TOKEN") or ""
    if not sandbox_token:
        raise HTTPException(status_code=503, detail="Sandbox API token not configured")
    supplied = _bearer_token(request.headers.get("x-sandbox-authorization", ""))
    if not supplied:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if not hmac.compare_digest(supplied, sandbox_token):
        raise HTTPException(status_code=401, detail="Invalid bearer token")

_AUTH = [Depends(_require_auth)]

# Track active bash processes so they can be killed on cancel
_active_procs = {}  # pid -> subprocess.Popen
_proc_lock = threading.Lock()

class BashReq(BaseModel):
    command: str
    work_dir: str = "/app"
    timeout: int = 120

class ReadReq(BaseModel):
    path: str
    offset: Optional[int] = None
    limit: Optional[int] = 2000

class WriteReq(BaseModel):
    path: str
    content: str

class EditReq(BaseModel):
    path: str
    old_str: str
    new_str: str
    replace_all: bool = False
    mode: str = "replace"

class ExistsReq(BaseModel):
    path: str

# ── Fuzzy matching & edit utilities (embedded) ──

UNICODE_MAP = {
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u00a0": " ", "\u2003": " ", "\u2002": " ",
    "\u200b": "", "\ufeff": "",
}

def _normalize_unicode(s):
    return "".join(UNICODE_MAP.get(c, c) for c in s)

def _fuzzy_find_original(content, pattern):
    """Find the original text in content that matches pattern fuzzily."""
    if pattern in content:
        return pattern, None
    # Pass 2: right-trim
    c_lines = content.split("\n")
    c_rt = "\n".join(l.rstrip() for l in c_lines)
    p_rt = "\n".join(l.rstrip() for l in pattern.split("\n"))
    if p_rt in c_rt:
        idx = c_rt.index(p_rt)
        start_line = c_rt[:idx].count("\n")
        n_lines = p_rt.count("\n") + 1
        matched = "\n".join(c_lines[start_line:start_line + n_lines])
        return matched, "(matched after trimming trailing whitespace)"
    # Pass 3: both-sides trim
    c_st = "\n".join(l.strip() for l in c_lines)
    p_st = "\n".join(l.strip() for l in pattern.split("\n"))
    if p_st in c_st:
        idx = c_st.index(p_st)
        start_line = c_st[:idx].count("\n")
        n_lines = p_st.count("\n") + 1
        matched = "\n".join(c_lines[start_line:start_line + n_lines])
        return matched, "(matched after trimming whitespace)"
    # Pass 4: unicode normalization
    c_norm = _normalize_unicode(c_st)
    p_norm = _normalize_unicode(p_st)
    if p_norm in c_norm:
        idx = c_norm.index(p_norm)
        start_line = c_norm[:idx].count("\n")
        n_lines = p_norm.count("\n") + 1
        matched = "\n".join(c_lines[start_line:start_line + n_lines])
        return matched, "(matched after unicode normalization)"
    return None, None

def _apply_edit(content, old_str, new_str, mode="replace", replace_all=False):
    """Apply edit. Returns (new_content, count, fuzzy_note) or raises ValueError."""
    if mode == "replace_all":
        replace_all = True
        mode = "replace"
    fuzzy_note = None
    if old_str not in content:
        matched, fuzzy_note = _fuzzy_find_original(content, old_str)
        if matched is None:
            raise ValueError("old_str not found in file.")
        old_str = matched
    count = content.count(old_str)
    if mode == "replace":
        if count > 1 and not replace_all:
            raise ValueError(f"old_str appears {count} times. Use replace_all=true or provide more context.")
        if replace_all:
            return content.replace(old_str, new_str), count, fuzzy_note
        return content.replace(old_str, new_str, 1), 1, fuzzy_note
    elif mode == "append_after":
        if replace_all:
            return content.replace(old_str, old_str + new_str), count, fuzzy_note
        idx = content.index(old_str) + len(old_str)
        return content[:idx] + new_str + content[idx:], 1, fuzzy_note
    elif mode == "prepend_before":
        if replace_all:
            return content.replace(old_str, new_str + old_str), count, fuzzy_note
        idx = content.index(old_str)
        return content[:idx] + new_str + content[idx:], 1, fuzzy_note
    raise ValueError(f"Unknown mode: {mode}")

def _validate_python(content, path=""):
    """Validate Python: syntax, kwargs against real installed signatures, training heuristics.

    Runs inside the sandbox where packages are pip-installed, so we can actually
    import classes and inspect their __init__ signatures to catch kwarg mismatches
    before runtime.
    """
    import ast as _ast, inspect as _inspect, importlib as _il
    warnings = []

    # 1. Syntax check
    try:
        tree = _ast.parse(content)
    except SyntaxError as e:
        warnings.append(f"Python syntax error at line {e.lineno}: {e.msg}")
        return warnings

    # 2. Build import map: name -> module path (from the script's own imports)
    import_map = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ImportFrom) and node.module:
            for alias in (node.names or []):
                local_name = alias.asname or alias.name
                import_map[local_name] = (node.module, alias.name)
        elif isinstance(node, _ast.Import):
            for alias in (node.names or []):
                local_name = alias.asname or alias.name
                import_map[local_name] = (alias.name, None)

    # 3. For each Call node, resolve the callable and check kwargs against signature
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        # Skip calls with **kwargs unpacking — we can't statically know those keys
        if any(kw.arg is None for kw in node.keywords):
            continue
        call_kwargs = [kw.arg for kw in node.keywords if kw.arg]
        if not call_kwargs:
            continue

        # Resolve the callable name
        func_name = None
        if isinstance(node.func, _ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, _ast.Attribute):
            func_name = node.func.attr
        if not func_name or func_name not in import_map:
            continue

        # Try to import and inspect the real callable
        module_path, attr_name = import_map[func_name]
        try:
            mod = _il.import_module(module_path)
            obj = getattr(mod, attr_name, None) if attr_name else mod
            if obj is None:
                continue
            sig = _inspect.signature(obj)
            params = sig.parameters
            # If **kwargs is in the signature, any kwarg is valid
            if any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values()):
                continue
            valid_names = set(params.keys())
            for kw_name in call_kwargs:
                if kw_name not in valid_names:
                    warnings.append(
                        f"Invalid kwarg: {func_name}({kw_name}=...) at line {node.lineno} "
                        f"-- not accepted by {module_path}.{attr_name or func_name}()"
                    )
        except Exception:
            pass  # can't import/inspect — skip silently

    # 4. Training script heuristics
    if any(kw in content for kw in ("TrainingArguments", "SFTConfig", "DPOConfig", "GRPOConfig")):
        if "push_to_hub" not in content:
            warnings.append("Training script warning: no 'push_to_hub' found")
        if "hub_model_id" not in content:
            warnings.append("Training script warning: no 'hub_model_id' found")
    return warnings

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/bash", dependencies=_AUTH)
def bash(req: BashReq):
    try:
        proc = subprocess.Popen(
            req.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=req.work_dir, start_new_session=True,
        )
        with _proc_lock:
            _active_procs[proc.pid] = proc
        try:
            stdout, stderr = proc.communicate(timeout=req.timeout)
            output = _strip_ansi(stdout + stderr)
            output = _truncate_output(output)
            return {"success": proc.returncode == 0, "output": output, "error": "" if proc.returncode == 0 else f"Exit code {proc.returncode}"}
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
            return {"success": False, "output": "", "error": f"Timeout after {req.timeout}s"}
        finally:
            with _proc_lock:
                _active_procs.pop(proc.pid, None)
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

@app.post("/api/kill", dependencies=_AUTH)
def kill_all():
    """Kill all active bash processes. Called when user cancels."""
    with _proc_lock:
        pids = list(_active_procs.keys())
    killed = []
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            killed.append(pid)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except OSError:
                pass
    return {"success": True, "output": f"Killed {len(killed)} process(es): {killed}", "error": ""}

@app.post("/api/read", dependencies=_AUTH)
def read(req: ReadReq):
    try:
        p = pathlib.Path(req.path)
        if not p.exists():
            return {"success": False, "output": "", "error": f"File not found: {req.path}"}
        if p.is_dir():
            return {"success": False, "output": "", "error": f"Is a directory: {req.path}"}
        lines = p.read_text().splitlines()
        start = (req.offset or 1) - 1
        end = start + (req.limit or len(lines))
        selected = lines[start:end]
        numbered = "\n".join(f"{start + i + 1}\t{line}" for i, line in enumerate(selected))
        return {"success": True, "output": numbered, "error": ""}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

@app.post("/api/write", dependencies=_AUTH)
def write(req: WriteReq):
    try:
        p = pathlib.Path(req.path)
        _atomic_write(p, req.content)
        msg = f"Wrote {len(req.content)} bytes to {req.path}"
        if p.suffix == ".py":
            warnings = _validate_python(req.content, req.path)
            if warnings:
                msg += "\n\nValidation warnings:\n" + "\n".join(f"  ! {w}" for w in warnings)
        return {"success": True, "output": msg, "error": ""}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

@app.post("/api/edit", dependencies=_AUTH)
def edit(req: EditReq):
    try:
        p = pathlib.Path(req.path)
        if not p.exists():
            return {"success": False, "output": "", "error": f"File not found: {req.path}"}
        content = p.read_text()
        if req.old_str == req.new_str:
            return {"success": False, "output": "", "error": "old_str and new_str must differ."}
        try:
            new_content, count, fuzzy_note = _apply_edit(
                content, req.old_str, req.new_str, mode=req.mode, replace_all=req.replace_all
            )
        except ValueError as e:
            return {"success": False, "output": "", "error": str(e)}
        _atomic_write(p, new_content)
        msg = f"Edited {req.path} ({count} replacement{'s' if count > 1 else ''})"
        if fuzzy_note:
            msg += f" {fuzzy_note}"
        if p.suffix == ".py":
            warnings = _validate_python(new_content, req.path)
            if warnings:
                msg += "\n\nValidation warnings:\n" + "\n".join(f"  ! {w}" for w in warnings)
        return {"success": True, "output": msg, "error": ""}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

@app.post("/api/exists", dependencies=_AUTH)
def exists(req: ExistsReq):
    return {"success": True, "output": str(pathlib.Path(req.path).exists()).lower(), "error": ""}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
