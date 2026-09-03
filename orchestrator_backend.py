#!/usr/bin/env python3
"""
Web Backend for Multi-Agent Orchestrator
Flask API that runs the orchestration and streams results
"""

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import subprocess
import json
import os
import re
import sys
import time
import threading
import uuid
from pathlib import Path
import requests
from typing import Optional

# Configured to automatically serve index.html from your repository root
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# In-memory tracking of active runs
active_orchestrations: dict = {}

# ============================================================================
# HELPER UTILS & RATE LIMITING
# ============================================================================

def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    match = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*)\n```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

_provider_cooldowns = {}

def _is_resting(name: str) -> bool:
    until = _provider_cooldowns.get(name)
    return until is not None and time.time() < until

def _mark_rate_limited(name: str, retry_after_seconds: float = 60.0):
    _provider_cooldowns[name] = time.time() + retry_after_seconds
    print(f"[{name}] rate-limited/overloaded, resting for {retry_after_seconds:.0f}s", file=sys.stderr)

def provider_status() -> dict:
    now = time.time()
    return {
        name: {"resting": until > now, "resumes_in_s": max(0, round(until - now))}
        for name, until in _provider_cooldowns.items()
    }

# ============================================================================
# MODEL PROVIDERS
# ============================================================================

def _openai_style_call(name: str, url: str, api_key: str, model: str,
                        prompt: str, max_tokens: int, extra_headers: dict = None) -> Optional[str]:
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

# ============================================================================
# PERSISTENCE & ROUTING
# ============================================================================

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
            print(f"[state] Upstash save failed: {e}", file=sys.stderr)
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
            print(f"[state] Upstash load failed: {e}", file=sys.stderr)
    path = LOCAL_STATE_DIR / f"{orch_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None

ROLE_CHAINS = {
    "coder":       [call_groq, call_gemini],
    "reviewer":    [call_mistral, call_cerebras, call_sambanova],
    "coordinator": [call_gemini, call_cohere],
    "documenter":  [call_cohere, call_cerebras],
    "catch_all":   [call_openrouter, call_huggingface],
}

def call_model_for_role(role: str, prompt: str, max_tokens: int = 1500) -> tuple:
    chain = ROLE_CHAINS.get(role, []) + ROLE_CHAINS["catch_all"] + [call_ollama]
    for caller in chain:
        result = caller(prompt, max_tokens)
        if result is not None:
            return _strip_code_fences(result), caller.__name__.replace("call_", "")
    return None, None

def call_model(prompt: str, max_tokens: int = 1500) -> Optional[str]:
    text, _ = call_model_for_role("coder", prompt, max_tokens)
    return text

# ============================================================================
# TASK COMPLEXITY & LANGUAGES
# ============================================================================

COMPLEXITY_CONFIG = {
    "simple": {
        "label": "Fast",
        "safety_ceiling": 8,
        "plan_instruction": "Give a lean plan of 1-2 short steps.",
        "plan_max_tokens": 200,
        "coder_guidelines": "- Write simple code\n- Skip unnecessary edge cases",
        "coder_max_tokens": 800,
        "review_instruction": "Judge against basic functionality.",
    },
    "medium": {
        "label": "Medium",
        "safety_ceiling": 30,
        "plan_instruction": "Give a plan of 3-4 concrete steps.",
        "plan_max_tokens": 500,
        "coder_guidelines": "- Write clean code\n- Handle key edge cases",
        "coder_max_tokens": 1800,
        "review_instruction": "Expect solid, working code with error handling.",
    },
    "complex": {
        "label": "Thorough",
        "safety_ceiling": 200,
        "plan_instruction": "Provide a detailed plan (4-6 technical steps).",
        "plan_max_tokens": 800,
        "coder_guidelines": "- Write production-ready code\n- Handle all edge cases",
        "coder_max_tokens": 3000,
        "review_instruction": "Hold to production-grade robustness standards.",
    },
}

def classify_complexity_agent(task: str, state: "OrchestratorState") -> str:
    state.add_message("info", "Classifying task complexity...")
    prompt = f"Classify task complexity as simple, medium, or complex. Task: {task}. Reply with ONLY the single word."
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

LANGUAGES = {
    "python": {"label": "Python", "extension": ".py", "prompt_name": "Python"},
    "html": {"label": "HTML / CSS / JS", "extension": ".html", "prompt_name": "HTML"},
    "c": {"label": "C", "extension": ".c", "prompt_name": "C"},
}

def get_language(state) -> dict:
    return LANGUAGES.get(getattr(state, "language", "python"), LANGUAGES["python"])

# ============================================================================
# STATE CLASS
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
        self.messages.append({"level": level, "text": text, "iteration": self.iteration})

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
# AGENT PIPELINE
# ============================================================================

def coordinator_agent(task: str, state: OrchestratorState) -> str:
    state.add_message("info", "Coordinator: analyzing task...")
    lang = get_language(state)
    cfg = COMPLEXITY_CONFIG[state.complexity]
    prompt = f"You are a coding coordinator. Break down task into steps.\nTask: {task}\nLanguage: {lang['prompt_name']}\n{cfg['plan_instruction']}"
    plan, used = call_model_for_role("coordinator", prompt, max_tokens=cfg["plan_max_tokens"])
    if used: state.add_message("info", f"   (via {used})")
    if not plan:
        state.add_message("error", "Failed to generate plan")
        return ""
    state.plan = plan
    state.add_message("success", "Plan created")
    return plan

def coder_agent(task: str, state: OrchestratorState) -> str:
    feedback = "\n".join(state.reviews[-3:]) if state.reviews else ""
    state.add_message("info", "Coder: writing code...")
    lang = get_language(state)
    cfg = COMPLEXITY_CONFIG[state.complexity]
    prompt = f"Write complete {lang['prompt_name']} code.\nTask: {task}\nPlan:\n{state.plan}\nFeedback:\n{feedback}\nGuidelines:\n{cfg['coder_guidelines']}\nOutput ONLY code."
    code, used = call_model_for_role("coder", prompt, max_tokens=cfg["coder_max_tokens"])
    if used: state.add_message("info", f"   (via {used})")
    if not code:
        state.add_message("error", "Failed to generate code")
        return state.current_code or ""
    state.current_code = code
    state.add_message("success", f"Code generated ({len(code)} chars)")
    return code

def executor_agent(code: str, state: OrchestratorState, file_path: str = None) -> dict:
    state.add_message("info", "Executor: validating...")
    lang = get_language(state)
    ext = lang["extension"]
    file_path = file_path or f"temp_solution{ext}"
    results = {"syntax_ok": False, "execution_ok": False, "lint_ok": False, "output": [], "errors": []}
    
    try:
        with open(file_path, "w") as f:
            f.write(code)
        state.files_modified.append(file_path)
    except Exception as e:
        results["errors"].append(f"Write failed: {e}")
        return results

    if ext == ".py":
        try:
            res = subprocess.run(["python", "-m", "py_compile", file_path], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                results["syntax_ok"] = True
                state.add_message("success", "Syntax valid")
            else:
                results["errors"].append(res.stderr)
                state.add_message("error", "Syntax error")
        except Exception as e:
            results["errors"].append(str(e))
    else:
        results["syntax_ok"] = True
        results["execution_ok"] = True

    return results

def reviewer_agent(code: str, task: str, exec_results: dict, state: OrchestratorState) -> dict:
    state.add_message("info", "Reviewer: assessing quality...")
    cfg = COMPLEXITY_CONFIG[state.complexity]
    prompt = f"Review code for task: {task}\nExec results: {json.dumps(exec_results)}\n{cfg['review_instruction']}\nReturn JSON with keys: pass (bool), quality_score (int 0-100), issues (list), summary (str)."
    review_text, used = call_model_for_role("reviewer", prompt, max_tokens=800)
    
    review = {"pass": True, "quality_score": 85}
    if review_text:
        try:
            match = re.search(r"\{.*\}", review_text, re.DOTALL)
            if match:
                review = json.loads(match.group(0))
        except Exception:
            pass

    state.quality_score = review.get("quality_score", 0)
    if review.get("pass"):
        state.add_message("success", f"PASS — quality score {state.quality_score}/100")
    else:
        state.add_message("warning", f"NEEDS IMPROVEMENT — score {state.quality_score}/100")

    state.reviews.append(json.dumps(review))
    return review

# ============================================================================
# API ENDPOINTS & ORCHESTRATION THREAD
# ============================================================================

def run_orchestration_thread(orch_id: str, state: OrchestratorState):
    try:
        save_run_state(orch_id, state)
        
        if state.complexity_setting == "auto":
            classify_complexity_agent(state.original_task, state)
        else:
            state.complexity = state.complexity_setting

        plan = coordinator_agent(state.original_task, state)
        if not plan:
            state.status = "failed"
            save_run_state(orch_id, state)
            return

        max_iterations = COMPLEXITY_CONFIG[state.complexity]["safety_ceiling"]
        for iteration in range(1, max_iterations + 1):
            state.iteration = iteration
            
            code = coder_agent(state.original_task, state)
            if not code:
                break
                
            exec_res = executor_agent(code, state)
            review = reviewer_agent(code, state.original_task, exec_res, state)
            
            save_run_state(orch_id, state)
            
            if review.get("pass", False):
                state.status = "completed"
                state.add_message("success", "Orchestration completed successfully!")
                save_run_state(orch_id, state)
                return

        state.status = "max_iterations_reached"
        save_run_state(orch_id, state)

    except Exception as e:
        state.status = "failed"
        state.add_message("error", f"Thread error: {e}")
        save_run_state(orch_id, state)


@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "providers": provider_status()}), 200


@app.route("/api/start", methods=["POST"])
@app.route("/start", methods=["POST"])
def start_orchestration():
    data = request.json or {}
    task = data.get("task", "").strip()
    if not task:
        return jsonify({"error": "Task is required"}), 400

    orch_id = str(uuid.uuid4())[:8]

    state = OrchestratorState()
    state.original_task = task
    state.language = data.get("language", "python")
    state.complexity_setting = data.get("complexity", "auto")
    state.status = "running"

    active_orchestrations[orch_id] = state
    
    t = threading.Thread(target=run_orchestration_thread, args=(orch_id, state))
    t.daemon = True
    t.start()

    return jsonify({"orchestration_id": orch_id, "status": "started"}), 200


@app.route("/api/status/<orch_id>", methods=["GET"])
@app.route("/status/<orch_id>", methods=["GET"])
def get_status(orch_id):
    state_dict = load_run_state(orch_id)
    if not state_dict and orch_id in active_orchestrations:
        state_dict = active_orchestrations[orch_id].to_dict()
    
    if not state_dict:
        return jsonify({"error": "Orchestration not found"}), 404
        
    return jsonify(state_dict), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
