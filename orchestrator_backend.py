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
# When a provider returns 429 (or 503/Retry-After), we remember the
# timestamp it becomes available again, so the chain skips it without
# wasting a request until then.
_provider_cooldowns = {}  # provider_name -> unix timestamp when available again

def _is_resting(name: str) -> bool:
    until = _provider_cooldowns.get(name)
    return until is not None and time.time() < until

def _mark_rate_limited(name: str, retry_after_seconds: float = 60.0):
    _provider_cooldowns[name] = time.time() + retry_after_seconds
    print(f"[{name}] rate-limited/overloaded, resting for {retry_after_seconds:.0f}s", file=sys.stderr)

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
            res_data = response.json()
            if "choices" in res_data and len(res_data["choices"]) > 0:
                return res_data["choices"][0]["message"]["content"]
            print(f"[{name}] unexpected response format: {res_data}", file=sys.stderr)
            return None
        if response.status_code in (429, 503):
            retry_after = response.headers.get("Retry-After")
            _mark_rate_limited(name, float(retry_after) if retry_after else 60.0)
        else:
            print(f"[{name}] HTTP {response.status_code}: {response.text[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[{name}] request failed: {e}", file=sys.stderr)
        return None


def call_groq(prompt, max_tokens):
    return _openai_style_call("groq", "https://api.groq.com/openai/v1/chat/completions",
                               os.environ.get("GROQ_API_KEY", "").strip(),
                               "llama-3.3-70b-versatile", prompt, max_tokens)

def call_gemini(prompt, max_tokens):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[gemini] skipped: no API key set", file=sys.stderr)
        return None
    if _is_resting("gemini"):
        return None
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens},
            },
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                parts = candidates[0]["content"]["parts"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
            print(f"[gemini] empty parts or blocked response: {data}", file=sys.stderr)
            return None
        if response.status_code in (429, 503):
            _mark_rate_limited("gemini", 60.0)
        else:
            print(f"[gemini] HTTP {response.status_code}: {response.text[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[gemini] request failed: {e}", file=sys.stderr)
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
            res_data = response.json()
            if "message" in res_data and "content" in res_data["message"] and len(res_data["message"]["content"]) > 0:
                return res_data["message"]["content"][0]["text"]
            print(f"[cohere] unexpected format: {res_data}", file=sys.stderr)
            return None
        if response.status_code in (429, 503):
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
        if response.status_code in (429, 503):
            _mark_rate_limited("huggingface", 60.0)
        return None
    except Exception as e:
        print(f"[huggingface] request failed: {e}", file=sys.stderr)
        return None

def call_ollama(prompt, max_tokens):
    # Avoid spending time trying to hit local Ollama on cloud hosts
    if os.environ.get("RENDER"):
        return None
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
        return None


# --- Persistent state store -------------------------------------------
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
ROLE_CHAINS = {
    "coder":       [call_groq, call_gemini],
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
# TASK COMPLEXITY
# ============================================================================

COMPLEXITY_CONFIG = {
    "simple": {
        "label": "Fast",
        "safety_ceiling": 8,
        "plan_instruction": "Give a lean plan of 1-2 short steps — this is a small, self-contained task, don't overthink it.",
        "plan_max_tokens": 200,
        "coder_guidelines": (
            "- Write the simplest correct code that does the job\n"
            "- Do NOT add extra features, config options, or abstractions that weren't asked for\n"
            "- Comments only where genuinely useful, not on every line\n"
            "- It's fine to skip edge cases that aren't implied by the task"
        ),
        "coder_max_tokens": 800,
        "review_instruction": (
            "This is a SIMPLE, small task. Judge it against what was actually asked, not against "
            "production/enterprise standards. Do not fail it for missing things like extensive error "
            "handling, logging, tests, or configurability unless the task specifically asked for them."
        ),
    },
    "medium": {
        "label": "Medium",
        "safety_ceiling": 30,
        "plan_instruction": "Give a plan of 3-4 steps — enough to be concrete without over-engineering.",
        "plan_max_tokens": 500,
        "coder_guidelines": (
            "- Write clean, correct, reasonably robust code\n"
            "- Handle the edge cases a user would realistically hit\n"
            "- Comment non-obvious logic"
        ),
        "coder_max_tokens": 1800,
        "review_instruction": (
            "This is a MEDIUM-complexity task. Expect solid, working code with reasonable error "
            "handling — not full production hardening."
        ),
    },
    "complex": {
        "label": "Thorough",
        "safety_ceiling": 200,
        "plan_instruction": "Provide a detailed plan (4-6 steps). Be specific and technical.",
        "plan_max_tokens": 800,
        "coder_guidelines": (
            "- Write complete, production-ready code\n"
            "- Add comprehensive comments\n"
            "- Handle ALL edge cases\n"
            "- Use best practices"
        ),
        "coder_max_tokens": 3000,
        "review_instruction": (
            "This is a COMPLEX task meant for production use. Hold it to a high bar: robustness, "
            "edge cases, and best practices all matter here."
        ),
    },
}

def classify_complexity_agent(task: str, state: "OrchestratorState") -> str:
    """One short, cheap call to tag the task's complexity. Falls back to
    'medium' if every provider is unavailable, so classification failure
    never blocks the run."""
    state.add_message("info", "Classifying task complexity...")
    prompt = f"""Classify the complexity of this coding task as exactly one word: simple, medium, or complex.

- simple: a small script, one function, a single well-known algorithm, a trivial utility
- medium: a multi-function program, basic app logic, moderate integration of a few pieces
- complex: a multi-component system, something with real state/concurrency/architecture concerns, or an explicit request for production-grade robustness

Task: {task}

Respond with ONLY the single word: simple, medium, or complex."""

    result, used = call_model_for_role("coordinator", prompt, max_tokens=10)
    tier = "medium"
    if result:
        cleaned = result.strip().lower()
        for candidate in ("simple", "medium", "complex"):
            if candidate in cleaned:
                tier = candidate
                break
    state.complexity = tier
    state.add_message("info", f"Detected complexity: {COMPLEXITY_CONFIG[tier]['label']}" + (f" (via {used})" if used else ""))
    return tier

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
        self.complexity_setting = "auto"
        self.complexity = "medium"
        self.plan = ""
        self.current_code = ""
        self.files_modified = []
        self.test_results = []
        self.reviews = []
        self.iteration = 0
        self.safety_ceiling = 30
        self.quality_score = 0
        self.status = "initializing"
        self.messages = []
        self.auto_resume = True
        self.overview_doc = ""

    def add_message(self, level: str, text: str):
        self.messages.append({
            "level": level,
            "text": text,
            "iteration": self.iteration
        })

    def to_dict(self):
        return {
            "task": self.original_task,
            "language": self.language,
            "complexity_setting": self.complexity_setting,
            "complexity": self.complexity,
            "complexity_label": COMPLEXITY_CONFIG.get(self.complexity, COMPLEXITY_CONFIG["medium"])["label"],
            "plan": self.plan,
            "code": self.current_code,
            "iteration": self.iteration,
            "quality_score": self.quality_score,
            "status": self.status,
            "messages": self.messages,
            "files_modified": self.files_modified,
            "auto_resume": self.auto_resume,
            "overview_doc": self.overview_doc,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

# ============================================================================
# AGENTS
# ============================================================================

def coordinator_agent(task: str, state: OrchestratorState) -> str:
    state.add_message("info", "Coordinator: analyzing task...")

    lang = get_language(state)
    cfg = COMPLEXITY_CONFIG[state.complexity]
    prompt = f"""
You are an expert coding coordinator. Break down this task into clear, actionable steps.

Task: {task}
Target language: {lang['prompt_name']}

{cfg['plan_instruction']} Format as a numbered list. Be specific and technical.
"""

    plan, used = call_model_for_role("coordinator", prompt, max_tokens=cfg["plan_max_tokens"])
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
    cfg = COMPLEXITY_CONFIG[state.complexity]
    prompt = f"""
You are an expert code writer. Write complete, working {lang['prompt_name']} code for this task.

Task: {task}
Target language: {lang['prompt_name']}

Plan to follow:
{state.plan}

{f'Code to improve:{chr(10)}{state.current_code}' if state.current_code else 'Write entirely new code.'}

{feedback}

Guidelines:
{cfg['coder_guidelines']}

Output ONLY the code, no explanations, no markdown code fences.
"""

    code, used = call_model_for_role("coder", prompt, max_tokens=cfg["coder_max_tokens"])
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
            capture_output=True, text=True, timeout=15
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
            [binary_path], capture_output=True, text=True, timeout=15
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
                while self.stack and self.stack[-1] != tag:
                    self.stack.pop()
                if self.stack:
                    self.stack.pop()

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
    cfg = COMPLEXITY_CONFIG[state.complexity]
    prompt = f"""
You are a code reviewer. Evaluate this code against what the task actually required.

{cfg['review_instruction']}

Task: {task}
Target language: {lang['prompt_name']}

Code quality: (assess correctness, robustness, best practices — calibrated to the complexity note above)
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
        state.add_message("success", f"PASS — quality score {state.quality_score}/100")
    else:
        state.add_message("warning", f"NEEDS IMPROVEMENT — score {state.quality_score}/100")

    state.reviews.append(json.dumps(review))
    return review
