# Deploying to Render + Provider Setup

## 1. Get free API keys (add as many as you want — all optional)
Each provider is independent. The app skips any provider whose key isn't
set, so you can start with just one (Groq is the easiest) and add more
later without touching code.

| Provider | Sign up at | Env var name |
|---|---|---|
| Groq | console.groq.com | `GROQ_API_KEY` |
| Gemini | aistudio.google.com/apikey | `GEMINI_API_KEY` |
| Mistral (Codestral) | console.mistral.ai | `MISTRAL_API_KEY` |
| Cerebras | cloud.cerebras.ai | `CEREBRAS_API_KEY` |
| SambaNova | cloud.sambanova.ai | `SAMBANOVA_API_KEY` |
| Cohere | dashboard.cohere.com | `COHERE_API_KEY` |
| OpenRouter | openrouter.ai/keys | `OPENROUTER_API_KEY` |
| HuggingFace | huggingface.co/settings/tokens | `HUGGINGFACE_API_KEY` |

Each is free sign-up, no credit card, "API Keys" section in their dashboard.

## 2. Get a free Upstash Redis (for state that survives sleep/restart)
1. Go to **console.upstash.com** → sign up free
2. Create a database → choose **Regional** (Global costs more)
3. On the database page, copy the **REST URL** and **REST TOKEN**
4. Set as env vars: `UPSTASH_REDIS_URL` and `UPSTASH_REDIS_TOKEN`

Without this, paused-run state is saved to a local file instead — fine for
local testing, but it'll be lost whenever Render's free tier spins down.

## 3. Choose a password
Since this becomes a public URL that can compile/run code, set:
- `APP_PASSWORD` — any password you choose. The whole app then requires
  HTTP Basic Auth (browser will prompt for username/password — any
  username works, only the password is checked).

## 4. Deploy to Render
1. Push this project to a GitHub repo (Render deploys from Git)
2. Go to **render.com** → New → **Blueprint** → connect your repo
   (it'll auto-detect `render.yaml`)
3. Render will ask you to fill in each env var listed in `render.yaml` —
   paste in whichever provider keys, your Upstash credentials, and your
   `APP_PASSWORD`
4. Click deploy. First build takes a few minutes.
5. You'll get a URL like `https://multi-agent-orchestrator.onrender.com`

## Important: don't change `--workers 1` in the Procfile
The app keeps track of in-progress runs in memory (plus checkpoints to
Redis/disk). Gunicorn's `--workers 1 --threads 8` setting means everything
runs in a single process with multiple threads, which is required for that
in-memory tracking to work correctly. Bumping `--workers` above 1 would
split traffic across separate processes that don't share memory, and
status polling would randomly fail to find in-progress runs.

## What to expect on the free tier
- First request after 15+ min idle takes 30-60 seconds (cold start) —
  the loading state in the UI will just sit there during this, that's normal
- A run can safely span a cold start/sleep, since state checkpoints after
  every iteration
- 750 free instance-hours/month across your Render workspace — plenty for
  personal use unless you're running it nonstop
