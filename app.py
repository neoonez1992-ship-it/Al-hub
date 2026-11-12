import os, requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")
BASE = "https://openrouter.ai/api/v1"

def auth():
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("На сервере не задан OPENROUTER_API_KEY")
    return {"Authorization": "Bearer " + key, "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("APP_URL", "https://ai-hub.example"),
            "X-Title": "AI HUB"}

@app.get("/")
def index():
    return send_from_directory("static", "index.html")

@app.get("/api/health")
def health():
    return jsonify(ok=True, key_configured=bool(os.environ.get("OPENROUTER_API_KEY", "").strip()))

@app.get("/api/models")
def models():
    try:
        r = requests.get(BASE + "/models", timeout=30)
        r.raise_for_status()
        free = []
        for m in r.json().get("data", []):
            p = m.get("pricing") or {}
            try:
                if float(p.get("prompt", 1)) == 0 and float(p.get("completion", 1)) == 0:
                    free.append({"id": m["id"], "name": m.get("name") or m["id"]})
            except (ValueError, TypeError):
                pass
        return jsonify(models=free[:250])
    except Exception:
        return jsonify(error="Не удалось загрузить модели OpenRouter"), 502

def ask(model, messages):
    r = requests.post(BASE + "/chat/completions", headers=auth(),
        json={"model": model, "messages": messages, "max_tokens": 900}, timeout=120)
    if not r.ok:
        try: msg = r.json().get("error", {}).get("message", r.text)
        except Exception: msg = r.text
        raise RuntimeError(str(msg)[:400])
    return r.json()["choices"][0]["message"].get("content", "")

@app.post("/api/chat")
def chat():
    try:
        b = request.get_json(force=True)
        model, prompt = str(b.get("model", "")).strip(), str(b.get("prompt", "")).strip()
        if not model or not prompt: return jsonify(error="Выбери модель и напиши запрос"), 400
        hist = b.get("history") or []
        msgs = [{"role": x["role"], "content": str(x["content"])[:10000]}
                for x in hist[-10:] if x.get("role") in ("user", "assistant") and "content" in x]
        msgs.append({"role": "user", "content": prompt[:10000]})
        return jsonify(answer=ask(model, msgs))
    except RuntimeError as e: return jsonify(error=str(e)), 400
    except Exception: return jsonify(error="Ошибка сервера или соединения с OpenRouter"), 502

@app.post("/api/compare")
def compare():
    try:
        b = request.get_json(force=True)
        prompt = str(b.get("prompt", "")).strip()
        chosen = [str(x) for x in (b.get("models") or [])[:3]]
        if not prompt or not chosen: return jsonify(error="Напиши запрос и выбери модели"), 400
        out = []
        for m in chosen:
            try: out.append({"model": m, "answer": ask(m, [{"role":"user","content":prompt[:9000]}])})
            except Exception as e: out.append({"model":m, "error":str(e)[:300]})
        return jsonify(results=out)
    except Exception: return jsonify(error="Ошибка сравнения"), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8765)))
