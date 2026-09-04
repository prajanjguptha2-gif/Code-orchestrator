#!/usr/bin/env python3
"""
Web Backend for Multi-Agent Orchestrator
Flask API that runs the orchestration and streams results
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import subprocess
import json
import os
import re
import sys
import time
import threading
import queue
from pathlib import Path
import requests
from typing import Optional

app = Flask(__name__)
CORS(app)

# In-memory tracking of in-progress orchestration runs, keyed by orchestration
# ID. Requires --workers 1 (see Procfile/render.yaml) since gunicorn worker
# *processes* don't share memory — only threads within this one process do.
active_orchestrations: dict = {}

# ============================================================================
# MODEL PROVIDERS
#
# Each role (coder, reviewer, coordinator, documenter) has its own ordered
# chain of providers. call_model_for_role() tries each in turn, skipping any
# provider that's currently marked as rate-limited, and falls through to the
# next. Ollama (local) is the guaranteed-available last resort for every
# role, since it doesn't need an API key or internet.
#
# A provider only needs its API key env var set to be used — missing keys
# are silently skipped, not treated as errors, so you can add providers one
# at a time.
# ============================================================================

def _strip_code_fences(text: str) -> str:
    """Models love wrapping output in ```json / ```python fences even when
    told not to. Strip them so downstream json.loads() and file-writing get
    clean content."""
    if not text:
        return text
    text = text.strip()
    match = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*)\n```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


# --- Per-provider rate-limit tracking -------------------------------------
# When a provider returns 429 (or a Retry-After header), we remember the
# timestamp it becomes available again, so the chain skips it without
# wasting a request until then.
_provider_cooldowns = {}  # provider_name -> unix timestamp when available again

def _is_resting(name: str) -> bool:
    until = _provider_cooldowns.get(name)
    return until is not None and time.time() < until

def _mark_rate_limited(name: str, retry_after_seconds: float = 60.0):
    _provider_cooldowns[name] = time.time() + retry_after_seconds
    print(f"[{name}] rate-limited, resting for {retry_after_seconds:.0f}s", file=sys.stderr)

def provider_status() -> dict:
    """For the overview doc / status endpoint: which providers are resting."""
    now = time.time()
    return {
        name: {"resting": until > now, "resumes_in_s": max(0, round(until - now))}
        for name, until in _provider_cooldowns.items()
    }


# --- Individual provider callers ------------------------------------------
# Each returns the raw text on success, or None on any failure (missing key,
# network error, bad response) so the chain can move to the next provider.
# A 429 additionally calls _mark_rate_limited() so we skip it next time too.

def _openai_style_call(name: str, url: str, api_key: str, model: str,
                        prompt: str, max_tokens: int, extra_headers: dict = None) -> Optional[str]:
    """Shared logic for any provider using an OpenAI-compatible chat
    completions endpoint (Groq, Mistral, Cerebras, OpenRouter, etc.)."""
    if not api_key:
        print(f"[{name}] skipped: no API key set", file=sys.stderr)
        return None
    if _is_resting(name):
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            _mark_rate_limited(name, float(retry_after) if retry_after else 60.0)
        else:
            print(f"[{name}] HTTP {response.status_code}: {response.text[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[{name}] request failed: {e}", file=sys.stderr)
        return None


def call_groq(prompt, max_tokens):
    # llama-3.3-70b-versatile was deprecated by Groq (returns HTTP 404
    # model_not_found). openai/gpt-oss-120b is their current recommended
    # general-purpose replacement.
    return _openai_style_call("groq", "https://api.groq.com/openai/v1/chat/completions",
                               os.environ.get("GROQ_API_KEY", "").strip(),
                               "openai/gpt-oss-120b", prompt, max_tokens)

def call_gemini(prompt, max_tokens):
    name = "gemini"
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(f"[{name}] skipped: no API key set", file=sys.stderr)
        return None
    if _is_resting(name):
        return None
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens},
            },
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                print(f"[{name}] no candidates returned: {json.dumps(data)[:200]}", file=sys.stderr)
                return None
            parts = (candidates[0].get("content") or {}).get("parts")
            if not parts:
                # Gemini can return a candidate with no content (e.g. a
                # safety/finish-reason block) instead of raising an error.
                # Treat this as "no result" rather than letting the
                # downstream KeyError masquerade as a generic crash.
                reason = candidates[0].get("finishReason", "unknown")
                print(f"[{name}] empty parts, finishReason={reason}", file=sys.stderr)
                return None
            return parts[0].get("text")
        if response.status_code == 429:
            _mark_rate_limited(name, 60.0)
        elif response.status_code == 503:
            # Google's own servers overloaded ("UNAVAILABLE") — not a quota
            # issue, but still worth a short cooldown so this shows as
            # "resting" instead of silently vanishing like a missing key.
            _mark_rate_limited(name, 20.0)
        else:
            print(f"[{name}] HTTP {response.status_code}: {response.text[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[{name}] request failed: {e}", file=sys.stderr)
        return None

def call_mistral(prompt, max_tokens):
    return _openai_style_call("mistral", "https://api.mistral.ai/v1/chat/completions",
                               os.environ.get("MISTRAL_API_KEY", "").strip(),
                               "codestral-latest", prompt, max_tokens)

def call_cerebras(prompt, max_tokens):
    return _openai_style_call("cerebras", "https://api.cerebras.ai/v1/chat/completions",
                               os.environ.get("CEREBRAS_API_KEY", "").strip(),
                               "llama3.3-70b", prompt, max_tokens)

def call_sambanova(prompt, max_tokens):
    return _openai_style_call("sambanova", "https://api.sambanova.ai/v1/chat/completions",
                               os.environ.get("SAMBANOVA_API_KEY", "").strip(),
                               "Meta-Llama-3.3-70B-Instruct", prompt, max_tokens)

def call_cohere(prompt, max_tokens):
    api_key = os.environ.get("COHERE_API_KEY", "").strip()
    if not api_key:
        print("[cohere] skipped: no API key set", file=sys.stderr)
        return None
    if _is_resting("cohere"):
        return None
    try:
        response = requests.post(
            "https://api.cohere.com/v2/chat",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "command-r7b-12-2024",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["message"]["content"][0]["text"]
        if response.status_code == 429:
            _mark_rate_limited("cohere", 60.0)
        else:
            print(f"[cohere] HTTP {response.status_code}: {response.text[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[cohere] request failed: {e}", file=sys.stderr)
        return None

def call_openrouter(prompt, max_tokens):
    return _openai_style_call("openrouter", "https://openrouter.ai/api/v1/chat/completions",
                               os.environ.get("OPENROUTER_API_KEY", "").strip(),
                               "deepseek/deepseek-chat:free", prompt, max_tokens)

def call_huggingface(prompt, max_tokens):
    api_key = os.environ.get("HUGGINGFACE_API_KEY", "").strip()
    if not api_key:
        print("[huggingface] skipped: no API key set", file=sys.stderr)
        return None
    if _is_resting("huggingface"):
        return None
    try:
        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        if response.status_code == 429:
            _mark_rate_limited("huggingface", 60.0)
        return None
    except Exception as e:
        print(f"[huggingface] request failed: {e}", file=sys.stderr)
        return None

def call_ollama(prompt, max_tokens):
    name = "ollama"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": os.environ.get("OLLAMA_MODEL", "deepseek-coder"),
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": max_tokens},
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json()["response"]
        return None
    except Exception:
        # Ollama not running — not an error worth logging loudly, it's an
        # expected state when the app is deployed to the cloud with no
        # local model available.
        return None


# --- Persistent state store -------------------------------------------
# Render's free tier wipes local disk on every sleep/restart/redeploy, so a
# paused run can't survive on a local JSON file alone. If UPSTASH_REDIS_URL
# and UPSTASH_REDIS_TOKEN are set, state is saved there instead (free tier,
# doesn't expire, tiny setup — see CLOUD_SETUP.md). Falls back to a local
# file automatically for local development, where disk persistence isn't a
# problem.

UPSTASH_URL = os.environ.get("UPSTASH_REDIS_URL", "").strip()
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_TOKEN", "").strip()
LOCAL_STATE_DIR = Path("orchestration_runs")

def save_run_state(orch_id: str, state: "OrchestratorState"):
    if UPSTASH_URL and UPSTASH_TOKEN:
        try:
            requests.post(
                f"{UPSTASH_URL}/set/orchrun:{orch_id}",
                headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
                data=state.to_json(),
                timeout=10,
            )
            return
        except Exception as e:
            print(f"[state] Upstash save failed, falling back to local file: {e}", file=sys.stderr)
    LOCAL_STATE_DIR.mkdir(exist_ok=True)
    (LOCAL_STATE_DIR / f"{orch_id}.json").write_text(state.to_json())

def load_run_state(orch_id: str) -> Optional[dict]:
    if UPSTASH_URL and UPSTASH_TOKEN:
        try:
            response = requests.get(
                f"{UPSTASH_URL}/get/orchrun:{orch_id}",
                headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
                timeout=10,
            )
            if response.status_code == 200:
                result = response.json().get("result")
                if result:
                    return json.loads(result)
        except Exception as e:
            print(f"[state] Upstash load failed, checking local file: {e}", file=sys.stderr)
    path = LOCAL_STATE_DIR / f"{orch_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


# --- Role → provider chain --------------------------------------------
# Ollama is appended to every chain as the guaranteed-available fallback.
# coder tries Mistral's Codestral first — it's a code-specialized model,
# a better fit for this role than a general-purpose chat model — then
# falls back to Groq, then Gemini.
ROLE_CHAINS = {
    "coder":       [call_mistral, call_groq, call_gemini],
    "reviewer":    [call_mistral, call_cerebras, call_sambanova],
    "coordinator": [call_gemini, call_cohere],
    "documenter":  [call_cohere, call_cerebras],
    "catch_all":   [call_openrouter, call_huggingface],
}

def call_model_for_role(role: str, prompt: str, max_tokens: int = 1500) -> tuple:
    """Try each provider in the role's chain in order, then catch_all, then
    Ollama as the last resort. Returns (text, provider_name_used) or
    (None, None) if every provider failed."""
    chain = ROLE_CHAINS.get(role, []) + ROLE_CHAINS["catch_all"] + [call_ollama]
    for caller in chain:
        result = caller(prompt, max_tokens)
        if result is not None:
            return _strip_code_fences(result), caller.__name__.replace("call_", "")
    return None, None

def call_model(prompt: str, max_tokens: int = 1500) -> Optional[str]:
    """Back-compat simple entry point (no role) — uses the coder chain."""
    text, _ = call_model_for_role("coder", prompt, max_tokens)
    return text

# ============================================================================
# COMPLEXITY TIERING
#
# One short, cheap classifier call tags each task simple/medium/complex
# before the coordinator runs. The tier then scales plan length, coder
# guidelines, reviewer strictness, and the safety ceiling — so a one-line
# script doesn't get bounced through the same production-grade bar as a
# real application. "Thorough" is deliberately identical to the old
# fixed behavior: nothing changes for tasks that actually need it.
# ============================================================================

COMPLEXITY_TIERS = {
    "fast": {
        "label": "Fast",
        "plan_instruction": "Provide a lean plan (1-2 steps). Format as a numbered list. Be specific and technical.",
        "coder_guidelines": (
            "- Write the simplest correct code that satisfies the task\n"
            "- Do not add extra features, abstractions, or config options that weren't asked for\n"
            "- Skip edge cases that aren't implied by the task"
        ),
        "reviewer_standard": (
            "Judge this code against exactly what was asked — do not fail it for "
            "missing logging, tests, configuration, or anything else it was never asked to include."
        ),
        "safety_ceiling": 8,
    },
    "medium": {
        "label": "Medium",
        "plan_instruction": "Provide a plan (3-4 steps). Format as a numbered list. Be specific and technical.",
        "coder_guidelines": (
            "- Write clean, reasonably robust code\n"
            "- Handle realistic edge cases a user might actually hit\n"
            "- Follow reasonable best practices without over-engineering"
        ),
        "reviewer_standard": (
            "Look for solid working code with reasonable error handling — "
            "not necessarily a full production bar."
        ),
        "safety_ceiling": 30,
    },
    "thorough": {
        "label": "Thorough",
        "plan_instruction": "Provide a detailed plan (4-6 steps). Format as a numbered list. Be specific and technical.",
        "coder_guidelines": (
            "- Write complete, production-ready code\n"
            "- Add comprehensive comments\n"
            "- Handle ALL edge cases\n"
            "- Use best practices"
        ),
        "reviewer_standard": (
            "Hold this to a full production bar — correctness, robustness, "
            "error handling, and best practices all matter."
        ),
        "safety_ceiling": 200,
    },
}

def classify_complexity(task: str) -> tuple:
    """One short, cheap model call (~10 output tokens) that tags the task
    fast/medium/thorough. Uses the coordinator chain (already configured
    and cheap) rather than requiring a dedicated key. Falls back to
    'medium' if every provider for this call fails, so a classifier
    outage never blocks the whole run."""
    prompt = f"""Classify this coding task's complexity as exactly one word: fast, medium, or thorough.

fast = a one-liner or trivial script (e.g. "add two numbers", "print hello world")
medium = a small program with some real logic (e.g. "a to-do list CLI", "a basic calculator with a menu")
thorough = a real application (e.g. "a REST API", "a game", anything with multiple files, persistence, or many features)

Task: {task}

Respond with ONLY one word: fast, medium, or thorough."""
    result, used = call_model_for_role("coordinator", prompt, max_tokens=10)
    if result:
        cleaned = result.strip().lower()
        for tier in ("fast", "medium", "thorough"):
            if tier in cleaned:
                return tier, used
    return "medium", None  # safe default if classification itself fails


# ============================================================================
# LANGUAGE CONFIG
# ============================================================================

LANGUAGES = {
    "python": {
        "label": "Python",
        "extension": ".py",
        "prompt_name": "Python",
    },
    "html": {
        "label": "HTML / CSS / JS",
        "extension": ".html",
        "prompt_name": "HTML (a single self-contained file, with CSS in a <style> tag and JS in a <script> tag)",
    },
    "c": {
        "label": "C",
        "extension": ".c",
        "prompt_name": "C",
    },
}

def get_language(state) -> dict:
    """Look up config for the state's language, defaulting to python."""
    return LANGUAGES.get(getattr(state, "language", "python"), LANGUAGES["python"])

# ============================================================================
# STATE
# ============================================================================

class OrchestratorState:
    def __init__(self):
        self.original_task = ""
        self.language = "python"
        self.plan = ""
        self.current_code = ""
        self.files_modified = []
        self.test_results = []
        self.reviews = []
        self.iteration = 0
        # Default safety ceiling; overwritten once the complexity classifier
        # (or a forced tier) runs — see COMPLEXITY_TIERS. Kept as a fallback
        # in case classification is ever bypassed.
        self.safety_ceiling = 200
        self.complexity = "auto"       # requested: auto / fast / medium / thorough
        self.complexity_tier = ""      # resolved tier, set once classified
        self.quality_score = 0
        self.status = "initializing"
        self.messages = []
        self.auto_resume = True   # per-run toggle — set False to require manual approval
        self.overview_doc = ""    # populated when the run pauses

    def add_message(self, level: str, text: str):
        """Add a timestamped message"""
        self.messages.append({
            "level": level,  # info, success, error, warning
            "text": text,
            "iteration": self.iteration
        })

    def to_dict(self):
        return {
            "task": self.original_task,
            "language": self.language,
            "plan": self.plan,
            "code": self.current_code,
            "iteration": self.iteration,
            "quality_score": self.quality_score,
            "status": self.status,
            "messages": self.messages,
            "files_modified": self.files_modified,
            "auto_resume": self.auto_resume,
            "overview_doc": self.overview_doc,
            "complexity": self.complexity,
            "complexity_tier": self.complexity_tier,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

# ============================================================================
# AGENTS
# ============================================================================

def coordinator_agent(task: str, state: OrchestratorState) -> str:
    state.add_message("info", "Coordinator: analyzing task...")

    lang = get_language(state)
    tier = COMPLEXITY_TIERS.get(state.complexity_tier, COMPLEXITY_TIERS["medium"])
    prompt = f"""
You are an expert coding coordinator. Break down this task into clear, actionable steps.

Task: {task}
Target language: {lang['prompt_name']}

{tier['plan_instruction']}
"""

    plan, used = call_model_for_role("coordinator", prompt, max_tokens=800)
    if used:
        state.add_message("info", f"   (via {used})")
    if not plan:
        state.add_message("error", "Failed to generate plan")
        return ""

    state.plan = plan
    state.add_message("success", "Plan created")
    return plan

def coder_agent(task: str, state: OrchestratorState) -> str:
    feedback = ""
    if state.reviews:
        feedback = "\n\nFeedback to address:\n" + "\n".join(state.reviews[-3:])

    state.add_message("info", "Coder: writing code...")

    lang = get_language(state)
    tier = COMPLEXITY_TIERS.get(state.complexity_tier, COMPLEXITY_TIERS["medium"])
    prompt = f"""
You are an expert code writer. Write complete, runnable code for this task.

Task: {task}
Target language: {lang['prompt_name']}

Plan to follow:
{state.plan}

{f'Code to improve:{chr(10)}{state.current_code}' if state.current_code else 'Write entirely new code.'}

{feedback}

Guidelines:
{tier['coder_guidelines']}

Output ONLY the code, no explanations, no markdown code fences.
"""

    code, used = call_model_for_role("coder", prompt, max_tokens=3000)
    if used:
        state.add_message("info", f"   (via {used})")
    if not code:
        state.add_message("error", "Failed to generate code")
        return state.current_code or ""

    state.current_code = code
    state.add_message("success", f"Code generated ({len(code)} chars)")
    return code

def executor_agent(code: str, state: OrchestratorState, file_path: str = None) -> dict:
    state.add_message("info", "Executor: running validation...")

    lang = get_language(state)
    ext = lang["extension"]
    if file_path is None:
        file_path = f"temp_solution{ext}"
    elif not file_path.endswith(ext):
        file_path = f"{Path(file_path).stem}{ext}"

    results = {
        "syntax_ok": False,
        "execution_ok": False,
        "lint_ok": False,
        "output": [],
        "errors": []
    }

    try:
        with open(file_path, "w") as f:
            f.write(code)
        state.files_modified.append(file_path)
    except Exception as e:
        results["errors"].append(f"File write failed: {e}")
        state.add_message("error", f"File write failed: {e}")
        return results

    if ext == ".py":
        _validate_python(file_path, state, results)
    elif ext == ".c":
        _validate_c(file_path, state, results)
    elif ext == ".html":
        _validate_html(file_path, state, results)
    else:
        # Unknown language: just confirm the file was written.
        results["syntax_ok"] = True
        results["execution_ok"] = True
        results["output"].append("File written (no automated validator for this language)")
        state.add_message("warning", "No validator for this language — skipping checks")

    return results


def _validate_python(file_path: str, state: OrchestratorState, results: dict):
    try:
        result = subprocess.run(
            ["python", "-m", "py_compile", file_path],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            results["syntax_ok"] = True
            results["output"].append("Syntax check passed")
            state.add_message("success", "Syntax valid")
        else:
            results["errors"].append(f"Syntax error: {result.stderr}")
            results["output"].append("Syntax error")
            state.add_message("error", "Syntax error")
            return
    except Exception as e:
        results["errors"].append(f"Syntax check error: {e}")
        return

    try:
        result = subprocess.run(
            ["python", file_path],
            capture_output=True, text=True, timeout=15,
            # Generated code frequently calls input() (e.g. "take 2 numbers
            # from the user"), but this process has no interactive stdin —
            # without feeding something in, input() immediately raises
            # EOFError and every such script fails validation regardless
            # of correctness. 20 lines of a generic numeric-and-text value
            # covers most simple prompts (numbers parse fine as int/float,
            # and it's also valid as a plain string for text prompts).
            input="1\n" * 20,
        )
        if result.returncode == 0:
            results["execution_ok"] = True
            results["output"].append("Execution successful")
            state.add_message("success", "Execution successful")
        else:
            results["errors"].append(f"Execution failed: {result.stderr[:200]}")
            results["output"].append("Execution failed")
            state.add_message("error", "Execution failed")
    except subprocess.TimeoutExpired:
        results["errors"].append("Execution timeout")
        results["output"].append("Timeout")
        state.add_message("error", "Timeout (>15s)")
    except Exception as e:
        results["errors"].append(f"Execution error: {e}")

def _validate_c(file_path: str, state: OrchestratorState, results: dict):
    binary_path = str(Path(file_path).with_suffix(".exe" if sys.platform == "win32" else ""))
    try:
        compile_result = subprocess.run(
            ["gcc", file_path, "-o", binary_path],
            capture_output=True, text=True, timeout=15
        )
    except FileNotFoundError:
        results["errors"].append("gcc not found — install a C compiler (e.g. MinGW on Windows) to validate C code")
        results["output"].append("No C compiler available")
        state.add_message("error", "gcc not found — can't compile C code")
        return
    except Exception as e:
        results["errors"].append(f"Compile check error: {e}")
        return

    if compile_result.returncode == 0:
        results["syntax_ok"] = True
        results["output"].append("Compiled successfully")
        state.add_message("success", "Compiles cleanly")
    else:
        results["errors"].append(f"Compile error: {compile_result.stderr[:300]}")
        results["output"].append("Compile error")
        state.add_message("error", "Compile error")
        return

    try:
        run_result = subprocess.run(
            [binary_path], capture_output=True, text=True, timeout=15,
            # Same reasoning as the Python validator: no interactive stdin
            # means scanf()/getchar() etc. would hang or fail without this.
            input="1\n" * 20,
        )
        if run_result.returncode == 0:
            results["execution_ok"] = True
            results["output"].append("Execution successful")
            state.add_message("success", "Execution successful")
        else:
            results["errors"].append(f"Execution failed (exit code {run_result.returncode}): {run_result.stderr[:200]}")
            results["output"].append("Execution failed")
            state.add_message("error", "Execution failed")
    except subprocess.TimeoutExpired:
        results["errors"].append("Execution timeout")
        results["output"].append("Timeout")
        state.add_message("error", "Timeout (>15s)")
    except Exception as e:
        results["errors"].append(f"Execution error: {e}")

def _validate_html(file_path: str, state: OrchestratorState, results: dict):
    """HTML doesn't 'execute' server-side, so validate structure instead:
    well-formed tags and a check that any <script> content at least parses
    as valid JS-like syntax via a lightweight brace/paren balance check."""
    from html.parser import HTMLParser

    class _Checker(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.errors = []
            self.void_tags = {"br", "hr", "img", "input", "meta", "link", "area",
                               "base", "col", "embed", "source", "track", "wbr"}

        def handle_starttag(self, tag, attrs):
            if tag not in self.void_tags:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if self.stack and self.stack[-1] == tag:
                self.stack.pop()
            elif tag in self.stack:
                # Mismatched but recoverable — pop until we find it
                while self.stack and self.stack[-1] != tag:
                    self.stack.pop()
                if self.stack:
                    self.stack.pop()
            # else: stray closing tag, ignore

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        checker = _Checker()
        checker.feed(content)

        if checker.stack:
            results["errors"].append(f"Unclosed tags: {', '.join(checker.stack)}")
            results["output"].append("Unclosed HTML tags")
            state.add_message("error", f"Unclosed tags: {', '.join(checker.stack)}")
            return

        results["syntax_ok"] = True
        results["output"].append("HTML structure valid")
        state.add_message("success", "HTML structure valid")

        # Basic brace balance check for embedded <script> JS
        import re
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", content, re.DOTALL | re.IGNORECASE)
        js_ok = True
        for script in scripts:
            if script.count("{") != script.count("}") or script.count("(") != script.count(")"):
                js_ok = False
        if js_ok:
            results["execution_ok"] = True
            results["output"].append("Embedded script braces balanced")
            state.add_message("success", "Embedded JS looks structurally sound")
        else:
            results["errors"].append("Unbalanced braces/parens in <script> content")
            results["output"].append("Script braces unbalanced")
            state.add_message("error", "Unbalanced braces in embedded script")
    except Exception as e:
        results["errors"].append(f"HTML validation error: {e}")
        state.add_message("error", f"Validation error: {e}")

def reviewer_agent(code: str, task: str, exec_results: dict, state: OrchestratorState) -> dict:
    state.add_message("info", "Reviewer: assessing quality...")

    lang = get_language(state)
    tier = COMPLEXITY_TIERS.get(state.complexity_tier, COMPLEXITY_TIERS["medium"])
    prompt = f"""
You are a code reviewer.

Task: {task}
Target language: {lang['prompt_name']}

Review standard: {tier['reviewer_standard']}

Execution results: {json.dumps(exec_results, indent=2)}

Respond in JSON format ONLY:
{{
  "pass": true/false,
  "quality_score": 0-100,
  "issues": [
    {{"severity": "critical/high/medium", "description": "issue"}}
  ],
  "summary": "one sentence"
}}

Output ONLY JSON.
"""

    review_text, used = call_model_for_role("reviewer", prompt, max_tokens=800)
    if used:
        state.add_message("info", f"   (via {used})")
    if not review_text:
        state.add_message("error", "Review failed")
        return {"pass": False, "quality_score": 0}

    try:
        review = json.loads(review_text)
    except Exception:
        # Model may have added stray text around the JSON — try to
        # extract the first {...} block before giving up.
        match = re.search(r"\{.*\}", review_text, re.DOTALL)
        if match:
            try:
                review = json.loads(match.group(0))
            except Exception:
                review = {"pass": False, "quality_score": 50, "summary": "Review parse error"}
                state.add_message("warning", f"Couldn't parse reviewer JSON: {review_text[:150]}")
        else:
            review = {"pass": False, "quality_score": 50, "summary": "Review parse error"}
            state.add_message("warning", f"Couldn't parse reviewer JSON: {review_text[:150]}")

    state.quality_score = review.get("quality_score", 0)

    if review.get("pass"):
        state.add_message("success", f"PASS — quality: {review.get('quality_score', 0)}/100")
    else:
        state.add_message("warning", f"NEEDS WORK — quality: {review.get('quality_score', 0)}/100")

    return review


def documenter_agent(code: str, task: str, state: "OrchestratorState") -> str:
    """Runs once the reviewer passes — writes a short docs/summary blurb
    for the finished code. Non-critical: if every documenter provider is
    unavailable, we just skip it rather than blocking on it."""
    state.add_message("info", "Documenter: writing summary...")
    lang = get_language(state)
    prompt = f"""
Write a short (5-8 sentence) plain-English summary of this {lang['prompt_name']} code for a README.
Explain what it does and how to use/run it. No code blocks, just prose.

Task it fulfills: {task}

Code:
{code}
"""
    doc_text, used = call_model_for_role("documenter", prompt, max_tokens=500)
    if doc_text is None:
        state.add_message("warning", "Documenter unavailable — skipping (non-critical)")
        return ""
    state.add_message("success", f"Summary written (via {used})")
    return doc_text.strip()


def generate_overview_doc(state: "OrchestratorState") -> str:
    """Builds the pause/handoff summary shown when every provider chain for
    the current step has run dry mid-loop."""
    lang = get_language(state)
    resting = provider_status()
    resting_lines = "\n".join(
        f"- {name}: resumes in ~{info['resumes_in_s']}s" for name, info in resting.items() if info["resting"]
    ) or "- (none currently tracked as rate-limited — likely a missing API key or a non-429 failure)"
    return f"""# Orchestration Paused — Overview

**Task:** {state.original_task}
**Language:** {lang['label']}
**Iterations completed:** {state.iteration}
**Best quality score so far:** {state.quality_score}/100

## What was tried
{state.plan or '(no plan generated yet)'}

## Why it paused
Every provider available for the current step is currently unavailable (rate-limited or unconfigured).

## Providers currently resting
{resting_lines}

## What happens next
{"This run will resume automatically once a provider's quota should have reset." if state.auto_resume else "This run is waiting for your approval — resume it manually when ready."}
"""

# ============================================================================
# ORCHESTRATION LOOP
# ============================================================================

def run_orchestration_loop(task: str, orchestration_id: str, language: str = "python",
                            auto_resume: bool = True, resumed_state: "OrchestratorState" = None,
                            complexity: str = "auto"):
    """Run the orchestration loop. No fixed iteration cap — keeps going
    until the reviewer passes, or every provider for the current step is
    exhausted (all rate-limited / unconfigured / failing), at which point
    it writes an overview doc and either auto-resumes later or pauses for
    approval, depending on state.auto_resume."""
    state = resumed_state or OrchestratorState()
    if resumed_state is None:
        state.original_task = task
        state.language = language if language in LANGUAGES else "python"
        state.auto_resume = auto_resume
        state.complexity = complexity if complexity in ("auto", "fast", "medium", "thorough") else "auto"
    state.status = "running"
    active_orchestrations[orchestration_id] = state

    if not state.complexity_tier:
        if state.complexity != "auto":
            state.complexity_tier = state.complexity
            state.add_message("info", f"Detected complexity: {COMPLEXITY_TIERS[state.complexity_tier]['label']} (forced)")
        else:
            tier, used = classify_complexity(task)
            state.complexity_tier = tier
            via = f" (via {used})" if used else ""
            state.add_message("info", f"Detected complexity: {COMPLEXITY_TIERS[tier]['label']}{via}")
        state.safety_ceiling = COMPLEXITY_TIERS[state.complexity_tier]["safety_ceiling"]

    if not state.plan:
        coordinator_agent(task, state)
        if not state.plan:
            state.status = "paused"
            state.overview_doc = generate_overview_doc(state)
            save_run_state(orchestration_id, state)
            state.add_message("warning", "Paused — coordinator has no available provider right now.")
            return

    while state.iteration < state.safety_ceiling:
        state.iteration += 1
        state.add_message("info", f"--- ITERATION {state.iteration} ---")

        code = coder_agent(task, state)
        if not code:
            # Every coder-chain provider (incl. catch-all + Ollama) failed.
            state.status = "paused"
            state.overview_doc = generate_overview_doc(state)
            save_run_state(orchestration_id, state)
            state.add_message("warning", "Paused — no coder provider available right now.")
            return

        state.current_code = code
        exec_results = executor_agent(code, state)
        review = reviewer_agent(code, task, exec_results, state)

        if review.get("pass") and exec_results["syntax_ok"] and exec_results["execution_ok"]:
            state.status = "success"
            state.add_message("success", "SUCCESS — code is production-ready.")
            doc = documenter_agent(code, task, state)
            if doc:
                state.overview_doc = doc
            save_run_state(orchestration_id, state)
            return

        save_run_state(orchestration_id, state)  # checkpoint after every iteration
        state.add_message("info", "Refining...")

    state.status = "safety_ceiling_reached"
    state.add_message("warning", f"Safety ceiling reached ({state.safety_ceiling} iterations). Best: {state.quality_score}/100")
    save_run_state(orchestration_id, state)

# ============================================================================
# AUTH
#
# Once this app is deployed publicly (Render), anyone with the URL could
# submit tasks that compile and run code on the server. If APP_PASSWORD is
# set, every /api/* route requires HTTP Basic Auth with that password
# (any username). If it's not set (e.g. local dev on localhost), auth is
# skipped entirely — nothing breaks your existing local workflow.
# ============================================================================

APP_PASSWORD = os.environ.get("APP_PASSWORD", "").strip()

@app.before_request
def _require_auth():
    if not APP_PASSWORD:
        return  # no password configured — auth disabled (local dev)
    if not request.path.startswith("/api/") and request.path != "/":
        return
    auth = request.authorization
    if not auth or auth.password != APP_PASSWORD:
        return Response(
            "Authentication required", 401,
            {"WWW-Authenticate": 'Basic realm="Orchestrator"'}
        )

# ============================================================================
# API ROUTES
# ============================================================================

@app.route("/api/health", methods=["GET"])
def health():
    """Check if at least one model provider is available for the coder role
    (the most important one — nothing works without it)."""
    configured = [
        name for name, key in [
            ("groq", os.environ.get("GROQ_API_KEY")),
            ("gemini", os.environ.get("GEMINI_API_KEY")),
            ("mistral", os.environ.get("MISTRAL_API_KEY")),
            ("cerebras", os.environ.get("CEREBRAS_API_KEY")),
            ("sambanova", os.environ.get("SAMBANOVA_API_KEY")),
            ("cohere", os.environ.get("COHERE_API_KEY")),
            ("openrouter", os.environ.get("OPENROUTER_API_KEY")),
            ("huggingface", os.environ.get("HUGGINGFACE_API_KEY")),
        ] if key
    ]
    ollama_up = False
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        ollama_up = True
    except Exception:
        pass
    if configured or ollama_up:
        return jsonify({
            "status": "ready",
            "cloud_providers_configured": configured,
            "ollama_available": ollama_up,
        })
    return jsonify({
        "status": "error",
        "message": "No provider API keys are set, and Ollama isn't running. "
                   "Set at least one provider key (see CLOUD_SETUP.md) or run Ollama locally."
    }), 503

@app.route("/api/languages", methods=["GET"])
def languages():
    """List supported target languages for the UI dropdown."""
    return jsonify({key: cfg["label"] for key, cfg in LANGUAGES.items()})

@app.route("/api/orchestrate", methods=["POST"])
def orchestrate():
    """Start a new orchestration"""
    data = request.json
    task = data.get("task", "").strip()
    language = data.get("language", "python").strip().lower()

    if not task:
        return jsonify({"error": "Task is required"}), 400

    if language not in LANGUAGES:
        return jsonify({"error": f"Unsupported language '{language}'. Choose from: {', '.join(LANGUAGES)}"}), 400

    auto_resume = bool(data.get("auto_resume", True))  # per-run toggle
    complexity = data.get("complexity", "auto").strip().lower()
    if complexity not in ("auto", "fast", "medium", "thorough"):
        complexity = "auto"

    # Generate unique ID
    import uuid
    orch_id = str(uuid.uuid4())[:8]

    # Run in background thread
    thread = threading.Thread(
        target=run_orchestration_loop,
        args=(task, orch_id, language, auto_resume, None, complexity)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"id": orch_id, "status": "started"}), 202

@app.route("/api/status/<orch_id>", methods=["GET"])
def status(orch_id):
    """Get status of an orchestration"""
    state = active_orchestrations.get(orch_id)
    if state is None:
        saved = load_run_state(orch_id)
        if saved is None:
            return jsonify({"error": "Orchestration not found"}), 404
        return jsonify(saved)
    return jsonify(state.to_dict())

@app.route("/api/resume/<orch_id>", methods=["POST"])
def resume(orch_id):
    """Manually resume a paused run (used when auto_resume is off for
    that run, or if you want to force a retry before its cooldown ends)."""
    state = active_orchestrations.get(orch_id)
    if state is None:
        saved = load_run_state(orch_id)
        if saved is None:
            return jsonify({"error": "Orchestration not found"}), 404
        state = _state_from_dict(saved)
    if state.status != "paused":
        return jsonify({"error": f"Run is not paused (status: {state.status})"}), 400

    thread = threading.Thread(
        target=run_orchestration_loop,
        args=(state.original_task, orch_id, state.language, state.auto_resume, state)
    )
    thread.daemon = True
    thread.start()
    return jsonify({"id": orch_id, "status": "resuming"}), 202

@app.route("/api/toggle-auto-resume/<orch_id>", methods=["POST"])
def toggle_auto_resume(orch_id):
    """Flip the per-run auto-resume setting."""
    state = active_orchestrations.get(orch_id)
    if state is None:
        return jsonify({"error": "Orchestration not found (or not in memory — reload won't preserve this toggle across a server restart)"}), 404
    state.auto_resume = bool(request.json.get("auto_resume", True))
    save_run_state(orch_id, state)
    return jsonify({"id": orch_id, "auto_resume": state.auto_resume})


def _state_from_dict(d: dict) -> "OrchestratorState":
    """Rebuild an OrchestratorState from a saved dict (for resuming after
    the app restarted and lost in-memory state)."""
    state = OrchestratorState()
    state.original_task = d.get("task", "")
    state.language = d.get("language", "python")
    state.plan = d.get("plan", "")
    state.current_code = d.get("code", "")
    state.iteration = d.get("iteration", 0)
    state.quality_score = d.get("quality_score", 0)
    state.status = d.get("status", "paused")
    state.messages = d.get("messages", [])
    state.files_modified = d.get("files_modified", [])
    state.auto_resume = d.get("auto_resume", True)
    state.overview_doc = d.get("overview_doc", "")
    state.complexity = d.get("complexity", "auto")
    state.complexity_tier = d.get("complexity_tier", "")
    if state.complexity_tier in COMPLEXITY_TIERS:
        state.safety_ceiling = COMPLEXITY_TIERS[state.complexity_tier]["safety_ceiling"]
    return state


def _auto_resume_watcher():
    """Background loop: every 30s, checks paused runs with auto_resume=True
    and resumes any whose blocking providers should have quota again."""
    while True:
        time.sleep(30)
        for orch_id, state in list(active_orchestrations.items()):
            if state.status == "paused" and state.auto_resume:
                still_resting = any(info["resting"] for info in provider_status().values())
                if not still_resting:
                    state.add_message("info", "Auto-resuming — provider quota should be available again.")
                    thread = threading.Thread(
                        target=run_orchestration_loop,
                        args=(state.original_task, orch_id, state.language, state.auto_resume, state)
                    )
                    thread.daemon = True
                    thread.start()

threading.Thread(target=_auto_resume_watcher, daemon=True).start()

@app.route("/api/result/<orch_id>", methods=["GET"])
def result(orch_id):
    """Get final result"""
    if orch_id not in active_orchestrations:
        return jsonify({"error": "Orchestration not found"}), 404

    state = active_orchestrations[orch_id]

    if state.status != "success":
        return jsonify({
            "status": state.status,
            "quality_score": state.quality_score,
            "code": state.current_code
        })

    return jsonify({
        "status": "success",
        "quality_score": state.quality_score,
        "code": state.current_code,
        "iterations": state.iteration,
        "cost": 0
    })

@app.route("/", methods=["GET"])
def index():
    """Serve the web UI"""
    return serve_ui()

def serve_ui():
    """Return the Aeon web UI: dark, warm-paper themed, amber/gold accent,
    dusty sage for success, muted rust for errors, no emoji (line icons
    instead), and no provider names surfaced anywhere."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Aeon</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@500;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #211d1a;
                --bg-soft: #262119;
                --card: #2a241f;
                --card-raised: #302a24;
                --ink: #f2ead9;
                --ink-dim: #c9beac;
                --ink-faint: #8c8272;
                --amber: #d9a441;
                --amber-soft: rgba(217, 164, 65, 0.16);
                --sage: #8fa88a;
                --sage-soft: rgba(143, 168, 138, 0.14);
                --rust: #b5573f;
                --rust-soft: rgba(181, 87, 63, 0.14);
                --hairline: rgba(242, 234, 217, 0.08);
            }

            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background-color: var(--bg);
                background-image:
                    radial-gradient(circle at 1px 1px, rgba(242,234,217,0.035) 1px, transparent 0);
                background-size: 3px 3px;
                color: var(--ink);
                min-height: 100vh;
                padding: 28px 20px;
            }

            .container {
                max-width: 1180px;
                margin: 0 auto;
            }

            .header {
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
                padding: 8px 4px 26px;
                border-bottom: 1px solid var(--hairline);
                margin-bottom: 28px;
            }
            .header .wordmark {
                font-family: 'Nunito', sans-serif;
                font-weight: 800;
                font-size: 2.4em;
                letter-spacing: 0.01em;
                color: var(--ink);
            }
            .header .wordmark span { color: var(--amber); }
            .header .tagline {
                font-size: 0.95em;
                color: var(--ink-faint);
            }

            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 22px;
                margin-bottom: 22px;
            }
            @media (max-width: 820px) {
                .grid { grid-template-columns: 1fr; }
            }

            .panel {
                background: var(--card);
                border-radius: 14px;
                padding: 22px;
                box-shadow: 0 12px 30px -18px rgba(0,0,0,0.55), 0 2px 6px rgba(0,0,0,0.2);
                display: flex;
                flex-direction: column;
            }

            .panel-title {
                display: flex;
                align-items: center;
                gap: 10px;
                font-family: 'Nunito', sans-serif;
                font-weight: 700;
                font-size: 1.15em;
                margin-bottom: 16px;
                color: var(--ink);
            }
            .panel-title svg { flex-shrink: 0; color: var(--amber); }

            label.field-label {
                font-size: 0.85em;
                font-weight: 600;
                color: var(--ink-dim);
                margin-bottom: 6px;
                display: block;
            }

            select, textarea {
                width: 100%;
                background: var(--bg-soft);
                border: 1px solid var(--hairline);
                border-radius: 10px;
                color: var(--ink);
                font-family: 'Inter', sans-serif;
                font-size: 0.95em;
                padding: 12px 14px;
            }
            select {
                margin-bottom: 16px;
                appearance: none;
            }
            textarea {
                flex: 1;
                min-height: 160px;
                resize: vertical;
                line-height: 1.5;
            }
            textarea:focus, select:focus {
                outline: none;
                border-color: var(--amber);
                box-shadow: 0 0 0 3px var(--amber-soft);
            }

            .toggle-row {
                display: flex;
                align-items: center;
                gap: 10px;
                margin: 16px 0 6px;
                font-size: 0.9em;
                color: var(--ink-dim);
            }
            .toggle-row input { accent-color: var(--amber); width: 16px; height: 16px; }

            .button-row {
                display: flex;
                gap: 10px;
                margin-top: 16px;
                flex-wrap: wrap;
            }
            button {
                border: none;
                border-radius: 10px;
                font-family: 'Nunito', sans-serif;
                font-weight: 700;
                font-size: 0.92em;
                padding: 12px 20px;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
            }
            .btn-primary {
                background: var(--amber);
                color: #26200f;
                flex: 1;
            }
            .btn-primary:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 8px 18px -8px rgba(217,164,65,0.5); }
            .btn-primary:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }
            .btn-secondary {
                background: var(--card-raised);
                color: var(--ink-dim);
            }
            .btn-secondary:hover { color: var(--ink); }
            .btn-resume {
                background: var(--sage);
                color: #1c2a1a;
                display: none;
            }

            .status-banner {
                margin-top: 16px;
                padding: 12px 14px;
                border-radius: 10px;
                font-size: 0.9em;
                font-weight: 600;
                display: none;
                align-items: center;
                gap: 10px;
            }
            .status-banner.show { display: flex; }
            .status-banner.running { background: var(--amber-soft); color: var(--amber); }
            .status-banner.success { background: var(--sage-soft); color: var(--sage); }
            .status-banner.warning { background: var(--amber-soft); color: var(--amber); }
            .status-banner.error   { background: var(--rust-soft); color: var(--rust); }

            .spinner {
                width: 13px; height: 13px;
                border: 2px solid currentColor;
                border-right-color: transparent;
                border-radius: 50%;
                animation: spin 0.7s linear infinite;
                flex-shrink: 0;
            }
            @keyframes spin { to { transform: rotate(360deg); } }

            .overview-doc {
                display: none;
                white-space: pre-wrap;
                background: var(--bg-soft);
                border: 1px solid var(--hairline);
                border-left: 3px solid var(--amber);
                border-radius: 10px;
                padding: 14px;
                margin-top: 12px;
                font-size: 0.85em;
                color: var(--ink-dim);
                line-height: 1.5;
                max-height: 220px;
                overflow-y: auto;
            }

            .log {
                background: var(--bg-soft);
                border: 1px solid var(--hairline);
                border-radius: 10px;
                padding: 14px;
                height: 380px;
                overflow-y: auto;
                font-size: 0.87em;
                line-height: 1.7;
            }
            .log-entry { display: flex; align-items: flex-start; gap: 8px; padding: 3px 0; }
            .log-entry svg { flex-shrink: 0; margin-top: 3px; }
            .log-entry.info svg { color: var(--ink-faint); }
            .log-entry.success svg { color: var(--sage); }
            .log-entry.warning svg { color: var(--amber); }
            .log-entry.error svg { color: var(--rust); }
            .log-entry.info span { color: var(--ink-dim); }
            .log-entry.success span { color: var(--sage); }
            .log-entry.warning span { color: var(--amber); }
            .log-entry.error span { color: var(--rust); }

            .code-output {
                flex: 1;
                min-height: 260px;
                background: var(--bg-soft);
                font-family: 'SF Mono', 'Consolas', monospace;
                font-size: 0.85em;
                color: var(--ink);
                line-height: 1.6;
            }

            .stats-row {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 12px;
            }
            @media (max-width: 480px) {
                .stats-row { grid-template-columns: repeat(2, 1fr); }
            }
            .stat {
                background: var(--card-raised);
                border-radius: 10px;
                padding: 16px;
                text-align: center;
            }
            .stat-value {
                font-family: 'Nunito', sans-serif;
                font-weight: 800;
                font-size: 1.8em;
                color: var(--amber);
            }
            .stat-label {
                font-size: 0.8em;
                color: var(--ink-faint);
                margin-top: 4px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <div class="wordmark">Ae<span>o</span>n</div>
                    <div class="tagline">Describe a task. Aeon plans it, writes it, checks it, and keeps refining until it's ready.</div>
                </div>
            </div>

            <div class="grid">
                <div class="panel">
                    <div class="panel-title">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
                        Your task
                    </div>
                    <label class="field-label" for="languageSelect">Language</label>
                    <select id="languageSelect">
                        <option value="python">Python</option>
                        <option value="html">HTML / CSS / JS</option>
                        <option value="c">C</option>
                    </select>
                    <label class="field-label" for="complexitySelect">Speed / complexity</label>
                    <select id="complexitySelect">
                        <option value="auto">Auto-detect</option>
                        <option value="fast">Fast</option>
                        <option value="medium">Medium</option>
                        <option value="thorough">Thorough</option>
                    </select>
                    <label class="field-label" for="taskInput">What should Aeon build?</label>
                    <textarea id="taskInput" placeholder="Describe the code you want to create..."></textarea>
                    <label class="toggle-row">
                        <input type="checkbox" id="autoResumeToggle" checked>
                        Resume automatically once a paused step becomes available again
                    </label>
                    <div class="button-row">
                        <button class="btn-primary" id="orchestrateBtn" onclick="startOrchestration()">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 3l14 9-14 9V3z"/></svg>
                            Start
                        </button>
                        <button class="btn-secondary" onclick="clearAll()">Clear</button>
                        <button class="btn-resume" id="resumeBtn" onclick="resumeOrchestration()">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 3l14 9-14 9V3z"/></svg>
                            Resume now
                        </button>
                    </div>
                    <div id="status" class="status-banner"></div>
                    <div id="overviewDoc" class="overview-doc"></div>
                </div>

                <div class="panel">
                    <div class="panel-title">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>
                        Progress
                    </div>
                    <div class="log" id="messages"></div>
                </div>
            </div>

            <div class="grid">
                <div class="panel">
                    <div class="panel-title">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M16 18l6-6-6-6M8 6l-6 6 6 6"/></svg>
                        Generated code
                    </div>
                    <textarea class="code-output" id="codeOutput" readonly></textarea>
                    <div class="button-row">
                        <button class="btn-secondary" onclick="copyCode()">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                            Copy to clipboard
                        </button>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-title">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9M13 17V5M8 17v-4"/></svg>
                        Statistics
                    </div>
                    <div class="stats-row">
                        <div class="stat">
                            <div class="stat-value" id="qualityScore">—</div>
                            <div class="stat-label">Quality score</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value" id="iterationCount">—</div>
                            <div class="stat-label">Iterations</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value" id="complexityStat">—</div>
                            <div class="stat-label">Complexity</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">Free</div>
                            <div class="stat-label">Cost</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentOrchId = null;
            let pollInterval = null;

            const ICONS = {
                info: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>',
                success: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>',
                warning: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4M12 17h.01M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>',
                error: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>'
            };

            async function startOrchestration() {
                const task = document.getElementById("taskInput").value.trim();
                const language = document.getElementById("languageSelect").value;
                const complexity = document.getElementById("complexitySelect").value;
                if (!task) {
                    alert("Please enter a task");
                    return;
                }

                try {
                    const health = await fetch("/api/health").then(r => r.json());
                    if (health.status !== "ready") {
                        alert("System not ready: " + health.message);
                        return;
                    }
                } catch (e) {
                    alert("Cannot connect to backend");
                    return;
                }

                document.getElementById("orchestrateBtn").disabled = true;
                clearMessages();
                document.getElementById("codeOutput").value = "";
                document.getElementById("qualityScore").textContent = "—";
                document.getElementById("iterationCount").textContent = "—";
                document.getElementById("complexityStat").textContent = "—";

                const auto_resume = document.getElementById("autoResumeToggle").checked;
                document.getElementById("overviewDoc").style.display = "none";
                document.getElementById("resumeBtn").style.display = "none";

                const response = await fetch("/api/orchestrate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ task, language, auto_resume, complexity })
                });

                const data = await response.json();
                currentOrchId = data.id;

                showStatus("running", `Orchestration started (ID: ${currentOrchId})`);
                addMessage("info", "Starting orchestration...");

                pollInterval = setInterval(pollStatus, 1000);
            }

            async function pollStatus() {
                if (!currentOrchId) return;

                try {
                    const response = await fetch(`/api/status/${currentOrchId}`);
                    const state = await response.json();

                    const messagesDiv = document.getElementById("messages");
                    const lastCount = messagesDiv.children.length;

                    if (state.messages.length > lastCount) {
                        state.messages.slice(lastCount).forEach(msg => {
                            addMessage(msg.level, msg.text);
                        });
                    }

                    if (state.quality_score) {
                        document.getElementById("qualityScore").textContent = state.quality_score + "/100";
                    }
                    if (state.iteration) {
                        document.getElementById("iterationCount").textContent = state.iteration;
                    }
                    if (state.complexity_tier) {
                        const label = state.complexity_tier.charAt(0).toUpperCase() + state.complexity_tier.slice(1);
                        document.getElementById("complexityStat").textContent = label;
                    }

                    if (state.code) {
                        document.getElementById("codeOutput").value = state.code;
                    }

                    if (state.status === "success") {
                        clearInterval(pollInterval);
                        showStatus("success", "Orchestration complete.");
                        document.getElementById("orchestrateBtn").disabled = false;
                    } else if (state.status === "safety_ceiling_reached") {
                        clearInterval(pollInterval);
                        showStatus("error", "Safety ceiling reached.");
                        document.getElementById("orchestrateBtn").disabled = false;
                    } else if (state.status === "paused") {
                        clearInterval(pollInterval);
                        const waitMsg = state.auto_resume
                            ? "Paused — will resume automatically once available."
                            : "Paused — waiting for your approval to continue.";
                        showStatus("warning", waitMsg);
                        document.getElementById("orchestrateBtn").disabled = false;
                        if (state.overview_doc) {
                            const docDiv = document.getElementById("overviewDoc");
                            docDiv.textContent = state.overview_doc;
                            docDiv.style.display = "block";
                        }
                        document.getElementById("resumeBtn").style.display = "inline-flex";
                    }
                } catch (e) {
                    console.error("Poll error:", e);
                }
            }

            async function resumeOrchestration() {
                if (!currentOrchId) return;
                document.getElementById("resumeBtn").style.display = "none";
                await fetch(`/api/resume/${currentOrchId}`, { method: "POST" });
                showStatus("running", "Resuming...");
                pollInterval = setInterval(pollStatus, 1000);
            }

            function addMessage(level, text) {
                const messagesDiv = document.getElementById("messages");
                const entry = document.createElement("div");
                entry.className = "log-entry " + level;
                const icon = ICONS[level] || ICONS.info;
                entry.innerHTML = icon + "<span></span>";
                entry.querySelector("span").textContent = text;
                messagesDiv.appendChild(entry);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function clearMessages() {
                document.getElementById("messages").innerHTML = "";
            }

            function showStatus(type, text) {
                const status = document.getElementById("status");
                status.className = "status-banner show " + type;
                if (type === "running") {
                    status.innerHTML = '<span class="spinner"></span><span></span>';
                    status.querySelector("span:last-child").textContent = text;
                } else {
                    status.innerHTML = (ICONS[type === "error" ? "error" : type === "success" ? "success" : "warning"] || "") + "<span></span>";
                    status.querySelector("span").textContent = text;
                }
            }

            function copyCode() {
                const code = document.getElementById("codeOutput");
                code.select();
                document.execCommand("copy");
            }

            function clearAll() {
                document.getElementById("taskInput").value = "";
                clearMessages();
                document.getElementById("codeOutput").value = "";
                document.getElementById("status").classList.remove("show");
                document.getElementById("orchestrateBtn").disabled = false;
                document.getElementById("resumeBtn").style.display = "none";
                document.getElementById("overviewDoc").style.display = "none";
                if (pollInterval) clearInterval(pollInterval);
            }
        </script>
    </body>
    </html>
    """
    return html, 200, {"Content-Type": "text/html"}

if __name__ == "__main__":
    print("Aeon — multi-agent code orchestrator backend")
    print("Starting Flask server on http://localhost:5000")
    print("Make sure Ollama is running: ollama serve")
    app.run(host="0.0.0.0", port=5000, debug=False)
