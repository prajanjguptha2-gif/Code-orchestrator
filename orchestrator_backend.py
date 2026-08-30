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
    if not api_key or _is_resting(name):
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
    return _openai_style_call("groq", "https://api.groq.com/openai/v1/chat/completions",
                               os.environ.get("GROQ_API_KEY", "").strip(),
                               "llama-3.3-70b-versatile", prompt, max_tokens)

def call_gemini(prompt, max_tokens):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or _is_resting("gemini"):
        return None
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens},
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        if response.status_code == 429:
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
    if not api_key or _is_resting("cohere"):
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
    if not api_key or _is_resting("huggingface"):
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
        # No hard cap — the loop runs until the reviewer passes or every
        # provider for the current step is exhausted (see run_orchestration_loop).
        # This is kept only as a sane safety ceiling against runaway loops.
        self.safety_ceiling = 200
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
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

# ============================================================================
# AGENTS
# ============================================================================

def coordinator_agent(task: str, state: OrchestratorState) -> str:
    state.add_message("info", "📋 Coordinator: Analyzing task...")
    
    lang = get_language(state)
    prompt = f"""
You are an expert coding coordinator. Break down this task into clear, actionable steps.

Task: {task}
Target language: {lang['prompt_name']}

Provide a detailed plan (4-6 steps). Format as a numbered list. Be specific and technical.
"""
    
    plan, used = call_model_for_role("coordinator", prompt, max_tokens=800)
    if used:
        state.add_message("info", f"   (via {used})")
    if not plan:
        state.add_message("error", "Failed to generate plan")
        return ""
    
    state.plan = plan
    state.add_message("success", f"✓ Plan created")
    return plan

def coder_agent(task: str, state: OrchestratorState) -> str:
    feedback = ""
    if state.reviews:
        feedback = "\n\nFeedback to address:\n" + "\n".join(state.reviews[-3:])
    
    state.add_message("info", "💻 Coder: Writing code...")
    
    lang = get_language(state)
    prompt = f"""
You are an expert code writer. Write complete, production-ready code for this task.

Task: {task}
Target language: {lang['prompt_name']}

Plan to follow:
{state.plan}

{f'Code to improve:{chr(10)}{state.current_code}' if state.current_code else 'Write entirely new code.'}

{feedback}

Guidelines:
- Write complete, runnable {lang['prompt_name']} code
- Add comprehensive comments
- Handle ALL edge cases
- Use best practices

Output ONLY the code, no explanations, no markdown code fences.
"""
    
    code, used = call_model_for_role("coder", prompt, max_tokens=3000)
    if used:
        state.add_message("info", f"   (via {used})")
    if not code:
        state.add_message("error", "Failed to generate code")
        return state.current_code or ""
    
    state.current_code = code
    state.add_message("success", f"✓ Code generated ({len(code)} chars)")
    return code

def executor_agent(code: str, state: OrchestratorState, file_path: str = None) -> dict:
    state.add_message("info", "⚙️  Executor: Running validation...")
    
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
        results["output"].append("✓ File written (no automated validator for this language)")
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
            results["output"].append("✓ Syntax check passed")
            state.add_message("success", "✓ Syntax valid")
        else:
            results["errors"].append(f"Syntax error: {result.stderr}")
            results["output"].append("✗ Syntax error")
            state.add_message("error", "✗ Syntax error")
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
            results["output"].append("✓ Execution successful")
            state.add_message("success", "✓ Execution successful")
        else:
            results["errors"].append(f"Execution failed: {result.stderr[:200]}")
            results["output"].append("✗ Execution failed")
            state.add_message("error", "✗ Execution failed")
    except subprocess.TimeoutExpired:
        results["errors"].append("Execution timeout")
        results["output"].append("✗ Timeout")
        state.add_message("error", "✗ Timeout (>15s)")
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
        results["output"].append("✗ No C compiler available")
        state.add_message("error", "✗ gcc not found — can't compile C code")
        return
    except Exception as e:
        results["errors"].append(f"Compile check error: {e}")
        return

    if compile_result.returncode == 0:
        results["syntax_ok"] = True
        results["output"].append("✓ Compiled successfully")
        state.add_message("success", "✓ Compiles cleanly")
    else:
        results["errors"].append(f"Compile error: {compile_result.stderr[:300]}")
        results["output"].append("✗ Compile error")
        state.add_message("error", "✗ Compile error")
        return

    try:
        run_result = subprocess.run(
            [binary_path], capture_output=True, text=True, timeout=15
        )
        if run_result.returncode == 0:
            results["execution_ok"] = True
            results["output"].append("✓ Execution successful")
            state.add_message("success", "✓ Execution successful")
        else:
            results["errors"].append(f"Execution failed (exit code {run_result.returncode}): {run_result.stderr[:200]}")
            results["output"].append("✗ Execution failed")
            state.add_message("error", "✗ Execution failed")
    except subprocess.TimeoutExpired:
        results["errors"].append("Execution timeout")
        results["output"].append("✗ Timeout")
        state.add_message("error", "✗ Timeout (>15s)")
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
            results["output"].append("✗ Unclosed HTML tags")
            state.add_message("error", f"✗ Unclosed tags: {', '.join(checker.stack)}")
            return

        results["syntax_ok"] = True
        results["output"].append("✓ HTML structure valid")
        state.add_message("success", "✓ HTML structure valid")

        # Basic brace balance check for embedded <script> JS
        import re
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", content, re.DOTALL | re.IGNORECASE)
        js_ok = True
        for script in scripts:
            if script.count("{") != script.count("}") or script.count("(") != script.count(")"):
                js_ok = False
        if js_ok:
            results["execution_ok"] = True
            results["output"].append("✓ Embedded script braces balanced")
            state.add_message("success", "✓ Embedded JS looks structurally sound")
        else:
            results["errors"].append("Unbalanced braces/parens in <script> content")
            results["output"].append("✗ Script braces unbalanced")
            state.add_message("error", "✗ Unbalanced braces in embedded script")
    except Exception as e:
        results["errors"].append(f"HTML validation error: {e}")
        state.add_message("error", f"✗ Validation error: {e}")

def reviewer_agent(code: str, task: str, exec_results: dict, state: OrchestratorState) -> dict:
    state.add_message("info", "🔍 Reviewer: Assessing quality...")
    
    lang = get_language(state)
    prompt = f"""
You are a strict code reviewer. Evaluate this code for production readiness.

Task: {task}
Target language: {lang['prompt_name']}

Code quality: (assess correctness, robustness, best practices)
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
                state.add_message("warning", f"⚠️ Couldn't parse reviewer JSON: {review_text[:150]}")
        else:
            review = {"pass": False, "quality_score": 50, "summary": "Review parse error"}
            state.add_message("warning", f"⚠️ Couldn't parse reviewer JSON: {review_text[:150]}")
    
    state.quality_score = review.get("quality_score", 0)
    
    if review.get("pass"):
        state.add_message("success", f"✅ PASS - Quality: {review.get('quality_score', 0)}/100")
    else:
        state.add_message("warning", f"❌ NEEDS WORK - Quality: {review.get('quality_score', 0)}/100")
    
    return review


def documenter_agent(code: str, task: str, state: "OrchestratorState") -> str:
    """Runs once the reviewer passes — writes a short docs/summary blurb
    for the finished code. Non-critical: if every documenter provider is
    unavailable, we just skip it rather than blocking on it."""
    state.add_message("info", "📝 Documenter: Writing summary...")
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
        state.add_message("warning", "⚠️ Documenter unavailable — skipping (non-critical)")
        return ""
    state.add_message("success", f"✓ Summary written (via {used})")
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
Every provider available for the current step is currently unavailable
(rate-limited or unconfigured).

## Providers currently resting
{resting_lines}

## What happens next
{"This run will resume automatically once a provider's quota should have reset." if state.auto_resume else "This run is waiting for your approval — resume it manually when ready."}
"""

# ============================================================================
# ORCHESTRATION LOOP
# ============================================================================

def run_orchestration_loop(task: str, orchestration_id: str, language: str = "python",
                            auto_resume: bool = True, resumed_state: "OrchestratorState" = None):
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
    state.status = "running"
    active_orchestrations[orchestration_id] = state

    if not state.plan:
        coordinator_agent(task, state)
        if not state.plan:
            state.status = "paused"
            state.overview_doc = generate_overview_doc(state)
            save_run_state(orchestration_id, state)
            state.add_message("warning", "⏸ Paused — coordinator has no available provider right now.")
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
            state.add_message("warning", "⏸ Paused — no coder provider available right now.")
            return

        state.current_code = code
        exec_results = executor_agent(code, state)
        review = reviewer_agent(code, task, exec_results, state)

        if review.get("pass") and exec_results["syntax_ok"] and exec_results["execution_ok"]:
            state.status = "success"
            state.add_message("success", "✅ SUCCESS! Code is production-ready.")
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

    # Generate unique ID
    import uuid
    orch_id = str(uuid.uuid4())[:8]
    
    # Run in background thread
    thread = threading.Thread(
        target=run_orchestration_loop,
        args=(task, orch_id, language, auto_resume)
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
                    state.add_message("info", "▶ Auto-resuming — provider quota should be available again.")
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
    """Return the HTML UI"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🤖 Multi-Agent Code Orchestrator</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }
            .content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                padding: 30px;
            }
            @media (max-width: 768px) {
                .content { grid-template-columns: 1fr; }
            }
            .section {
                display: flex;
                flex-direction: column;
            }
            .section h2 {
                font-size: 1.5em;
                margin-bottom: 20px;
                color: #333;
            }
            textarea {
                flex: 1;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-family: "Monaco", "Courier New", monospace;
                font-size: 0.95em;
                resize: vertical;
                min-height: 200px;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            .messages {
                background: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                height: 400px;
                overflow-y: auto;
                font-family: "Monaco", "Courier New", monospace;
                font-size: 0.9em;
            }
            .message {
                padding: 8px 0;
                line-height: 1.5;
            }
            .message.info { color: #666; }
            .message.success { color: #28a745; }
            .message.error { color: #dc3545; }
            .message.warning { color: #ffc107; }
            .button-group {
                display: flex;
                gap: 10px;
                margin-top: 20px;
            }
            button {
                flex: 1;
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            .btn-primary:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }
            .btn-secondary {
                background: #e0e0e0;
                color: #333;
            }
            .btn-secondary:hover {
                background: #d0d0d0;
            }
            .status {
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                font-weight: 600;
                margin-top: 20px;
                display: none;
            }
            .status.show { display: block; }
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .status.running {
                background: #cfe2ff;
                color: #084298;
                border: 1px solid #b6d4fe;
            }
            .status.warning {
                background: #fff3cd;
                color: #664d03;
                border: 1px solid #ffe69c;
            }
            .spinner {
                display: inline-block;
                width: 12px;
                height: 12px;
                border: 2px solid currentColor;
                border-right-color: transparent;
                border-radius: 50%;
                animation: spin 0.6s linear infinite;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-top: 20px;
            }
            .stat {
                background: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-value {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Multi-Agent Code Orchestrator</h1>
                <p>Multi-provider (Groq/Gemini/Mistral/Cerebras/SambaNova/Cohere/OpenRouter) + Ollama fallback — Python · HTML · C</p>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2>Your Task</h2>
                    <label for="languageSelect" style="font-weight:600; margin-bottom:6px; display:block;">Language</label>
                    <select id="languageSelect" style="padding:10px; border:2px solid #e0e0e0; border-radius:8px; margin-bottom:15px; font-size:0.95em;">
                        <option value="python">Python</option>
                        <option value="html">HTML / CSS / JS</option>
                        <option value="c">C</option>
                    </select>
                    <textarea id="taskInput" placeholder="Describe the code you want to create..."></textarea>
                    <label style="display:flex; align-items:center; gap:8px; margin:10px 0; font-weight:500;">
                        <input type="checkbox" id="autoResumeToggle" checked>
                        Auto-resume when a rate limit clears (uncheck to require your approval instead)
                    </label>
                    <div class="button-group">
                        <button class="btn-primary" id="orchestrateBtn" onclick="startOrchestration()">
                            Start Orchestration
                        </button>
                        <button class="btn-secondary" onclick="clearAll()">Clear</button>
                        <button class="btn-secondary" id="resumeBtn" style="display:none;" onclick="resumeOrchestration()">
                            ▶ Resume Now
                        </button>
                    </div>
                    <div id="status" class="status"></div>
                    <div id="overviewDoc" style="display:none; white-space:pre-wrap; background:#fff8e1; border:1px solid #ffe082; border-radius:8px; padding:12px; margin-top:10px; font-size:0.9em;"></div>
                </div>
                
                <div class="section">
                    <h2>Live Messages</h2>
                    <div class="messages" id="messages"></div>
                </div>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2>Generated Code</h2>
                    <textarea id="codeOutput" readonly></textarea>
                    <button class="btn-primary" style="margin-top: 10px;" onclick="copyCode()">Copy to Clipboard</button>
                </div>
                
                <div class="section">
                    <h2>Statistics</h2>
                    <div class="stats">
                        <div class="stat">
                            <div class="stat-value" id="qualityScore">-</div>
                            <div class="stat-label">Quality Score</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value" id="iterationCount">-</div>
                            <div class="stat-label">Iterations</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">$0</div>
                            <div class="stat-label">Cost (FREE)</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentOrchId = null;
            let pollInterval = null;
            
            async function startOrchestration() {
                const task = document.getElementById("taskInput").value.trim();
                const language = document.getElementById("languageSelect").value;
                if (!task) {
                    alert("Please enter a task");
                    return;
                }
                
                // Check health
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
                document.getElementById("qualityScore").textContent = "-";
                document.getElementById("iterationCount").textContent = "-";
                
                const auto_resume = document.getElementById("autoResumeToggle").checked;
                document.getElementById("overviewDoc").style.display = "none";
                document.getElementById("resumeBtn").style.display = "none";

                const response = await fetch("/api/orchestrate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ task, language, auto_resume })
                });
                
                const data = await response.json();
                currentOrchId = data.id;
                
                showStatus("running", `Orchestration started (ID: ${currentOrchId})`);
                addMessage("info", "🚀 Starting orchestration...");
                
                // Poll for updates
                pollInterval = setInterval(pollStatus, 1000);
            }
            
            async function pollStatus() {
                if (!currentOrchId) return;
                
                try {
                    const response = await fetch(`/api/status/${currentOrchId}`);
                    const state = await response.json();
                    
                    // Update messages
                    const messagesDiv = document.getElementById("messages");
                    const lastCount = messagesDiv.children.length;
                    
                    if (state.messages.length > lastCount) {
                        state.messages.slice(lastCount).forEach(msg => {
                            addMessage(msg.level, msg.text);
                        });
                    }
                    
                    // Update stats
                    if (state.quality_score) {
                        document.getElementById("qualityScore").textContent = state.quality_score + "/100";
                    }
                    if (state.iteration) {
                        document.getElementById("iterationCount").textContent = state.iteration;
                    }
                    
                    // Update code
                    if (state.code) {
                        document.getElementById("codeOutput").value = state.code;
                    }
                    
                    // Check if done
                    if (state.status === "success") {
                        clearInterval(pollInterval);
                        showStatus("success", "✅ Orchestration complete!");
                        document.getElementById("orchestrateBtn").disabled = false;
                    } else if (state.status === "safety_ceiling_reached") {
                        clearInterval(pollInterval);
                        showStatus("error", "Safety ceiling reached");
                        document.getElementById("orchestrateBtn").disabled = false;
                    } else if (state.status === "paused") {
                        clearInterval(pollInterval);
                        const waitMsg = state.auto_resume
                            ? "⏸ Paused — will auto-resume once provider quota resets."
                            : "⏸ Paused — waiting for your approval to continue.";
                        showStatus("warning", waitMsg);
                        document.getElementById("orchestrateBtn").disabled = false;
                        if (state.overview_doc) {
                            const docDiv = document.getElementById("overviewDoc");
                            docDiv.textContent = state.overview_doc;
                            docDiv.style.display = "block";
                        }
                        document.getElementById("resumeBtn").style.display = "inline-block";
                    }
                } catch (e) {
                    console.error("Poll error:", e);
                }
            }

            async function resumeOrchestration() {
                if (!currentOrchId) return;
                document.getElementById("resumeBtn").style.display = "none";
                await fetch(`/api/resume/${currentOrchId}`, { method: "POST" });
                showStatus("running", "▶ Resuming...");
                pollInterval = setInterval(pollStatus, 1000);
            }
            
            function addMessage(level, text) {
                const messagesDiv = document.getElementById("messages");
                const msg = document.createElement("div");
                msg.className = "message " + level;
                msg.textContent = text;
                messagesDiv.appendChild(msg);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            function clearMessages() {
                document.getElementById("messages").innerHTML = "";
            }
            
            function showStatus(type, text) {
                const status = document.getElementById("status");
                status.className = "status show " + type;
                if (type === "running") {
                    status.innerHTML = `<span class="spinner"></span> ${text}`;
                } else {
                    status.textContent = text;
                }
            }
            
            function copyCode() {
                const code = document.getElementById("codeOutput");
                code.select();
                document.execCommand("copy");
                alert("Code copied to clipboard!");
            }
            
            function clearAll() {
                document.getElementById("taskInput").value = "";
                clearMessages();
                document.getElementById("codeOutput").value = "";
                document.getElementById("status").classList.remove("show");
                document.getElementById("orchestrateBtn").disabled = false;
                if (pollInterval) clearInterval(pollInterval);
            }
        </script>
    </body>
    </html>
    """
    return html, 200, {"Content-Type": "text/html"}

if __name__ == "__main__":
    print("🚀 Multi-Agent Orchestrator Backend")
    print("Starting Flask server on http://localhost:5000")
    print("Make sure Ollama is running: ollama serve")
    app.run(host="0.0.0.0", port=5000, debug=False)
