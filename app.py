import os
import base64
import mimetypes
import re
from flask import Flask, jsonify, request, send_from_directory
import requests

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

BASE = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "AI HUB")
TIMEOUT = 180
_cache = {"models": None, "images": None}

def headers(json_content=True):
    h = {"Authorization": f"Bearer {API_KEY}", "HTTP-Referer": SITE_URL or "https://ai-hub.local", "X-Title": SITE_NAME}
    if json_content:
        h["Content-Type"] = "application/json"
    return h

def require_key():
    if not API_KEY:
        return jsonify({"error": "На сервере не задана переменная OPENROUTER_API_KEY. Добавь её в Environment на Render."}), 500
    return None

def upstream(method, path, **kwargs):
    r = requests.request(method, BASE + path, headers=headers(), timeout=TIMEOUT, **kwargs)
    try:
        data = r.json()
    except Exception:
        data = {"error": r.text[:1000]}
    if not r.ok:
        msg = data.get("error", data)
        if isinstance(msg, dict):
            msg = msg.get("message", str(msg))
        raise RuntimeError(str(msg))
    return data

def get_models():
    if _cache["models"] is None:
        _cache["models"] = upstream("GET", "/models").get("data", [])
    return _cache["models"]

def get_image_models():
    if _cache["images"] is None:
        _cache["images"] = upstream("GET", "/images/models").get("data", [])
    return _cache["images"]

def is_vision(m):
    return "image" in (m.get("architecture") or {}).get("input_modalities", [])

def model_text(m):
    return " ".join([str(m.get("id", "")), str(m.get("name", "")), str(m.get("description", ""))]).lower()

def choose_model(prompt, has_images=False, requested="auto"):
    catalog = get_models()
    if requested and requested != "auto":
        chosen = next((m for m in catalog if m.get("id") == requested), None)
        if not chosen:
            raise RuntimeError("Выбранная модель отсутствует в актуальном каталоге OpenRouter.")
        if has_images and not is_vision(chosen):
            raise RuntimeError("Эта модель не поддерживает входящие изображения. Выбери модель с поддержкой фото или включи автоматический режим.")
        return chosen
    text = (prompt or "").lower()
    code = bool(re.search(r"код|программ|python|javascript|typescript|html|css|ошибк|debug|разработ|функци|скрипт|приложен", text))
    reasoning = bool(re.search(r"докажи|почему|сравни|проанализ|сложн|математ|логик|рассчитай|стратег|архитектур", text))
    candidates = [m for m in catalog if not has_images or is_vision(m)]
    if not candidates:
        raise RuntimeError("В каталоге не найдено моделей с поддержкой изображений.")
    def score(m):
        s = model_text(m)
        points = 0
        if code:
            if re.search(r"coder|codestral|code|deepseek", s): points += 8
        elif reasoning:
            if re.search(r"reason|thinking|o1|o3|r1|qwq|opus|sonnet", s): points += 7
        else:
            if re.search(r"sonnet|gpt|gemini|claude|qwen", s): points += 4
        if has_images and is_vision(m): points += 10
        pricing = m.get("pricing") or {}
        # Mild preference for free models, but not enough to override task suitability.
        try:
            if float(pricing.get("prompt", 1)) == 0 and float(pricing.get("completion", 1)) == 0: points += 1.5
        except (TypeError, ValueError):
            pass
        if m.get("context_length", 0) >= 32000: points += 0.5
        return points
    candidates.sort(key=score, reverse=True)
    return candidates[0]

def content_text(response):
    choices = response.get("choices") or []
    if choices:
        content = (choices[0].get("message") or {}).get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return response.get("output_text", "")

def extract_images(response):
    out = []
    for choice in response.get("choices", []):
        msg = choice.get("message", {})
        for part in (msg.get("images") or []):
            url = part.get("image_url", {}).get("url") or part.get("url")
            if url:
                out.append(url)
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    url = part.get("image_url", {}).get("url") if isinstance(part.get("image_url"), dict) else part.get("url")
                    if url:
                        out.append(url)
    return out

@app.get("/")
def index():
    return send_from_directory("static", "index.html")

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "configured": bool(API_KEY), "service": "AI HUB"})

@app.get("/api/models")
def models():
    err = require_key()
    if err: return err
    try:
        return jsonify({"data": get_models()})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.get("/api/images/models")
def image_models():
    err = require_key()
    if err: return err
    try:
        return jsonify({"data": get_image_models()})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.post("/api/chat")
def chat():
    err = require_key()
    if err: return err
    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt") or "")
        messages = data.get("messages") or [{"role": "user", "content": prompt}]
        last = messages[-1] if messages else {}
        parts = last.get("content")
        has_images = isinstance(parts, list) and any(p.get("type") == "image_url" for p in parts if isinstance(p, dict))
        chosen = choose_model(prompt, has_images, data.get("model", "auto") if data.get("mode") == "manual" else "auto")
        payload = {"model": chosen["id"], "messages": messages, "max_tokens": 3000}
        result = upstream("POST", "/chat/completions", json=payload)
        return jsonify({"selected_model": chosen["id"], "reply": content_text(result), "images": extract_images(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.post("/api/compare")
def compare():
    err = require_key()
    if err: return err
    data = request.get_json(force=True) or {}
    prompt = str(data.get("prompt") or "")
    ids = data.get("models") or []
    if not prompt or not ids:
        return jsonify({"error": "Нужен текст запроса и хотя бы одна модель."}), 400
    results = []
    for mid in ids[:3]:
        try:
            result = upstream("POST", "/chat/completions", json={"model": mid, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000})
            results.append({"model": mid, "reply": content_text(result)})
        except Exception as e:
            results.append({"model": mid, "error": str(e)})
    return jsonify({"results": results})

@app.post("/api/images/generate")
def generate_image():
    err = require_key()
    if err: return err
    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt") or "").strip()
        if not prompt: return jsonify({"error": "Напиши описание изображения."}), 400
        catalog = get_image_models()
        requested = data.get("model", "auto")
        if requested == "auto":
            # Choose a listed image generation model; prefer models with image output and a clear current catalog entry.
            chosen = catalog[0] if catalog else None
        else:
            chosen = next((m for m in catalog if m.get("id") == requested), None)
        if not chosen:
            return jsonify({"error": "Не найдена подходящая модель генерации изображений."}), 400
        payload = {"model": chosen["id"], "prompt": prompt, "aspect_ratio": data.get("aspect_ratio", "1:1"), "resolution": data.get("resolution", "1K")}
        result = upstream("POST", "/images", json=payload)
        images = []
        for item in result.get("data", []):
            if item.get("b64_json"):
                media = item.get("media_type") or "image/png"
                images.append("data:" + media + ";base64," + item["b64_json"])
            elif item.get("url"):
                images.append(item["url"])
        usage = result.get("usage") or {}
        return jsonify({"model": chosen["id"], "images": images, "cost": usage.get("cost"), "message": "Готово" if images else "Провайдер не вернул изображение."})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.post("/api/images/edit")
def edit_image():
    err = require_key()
    if err: return err
    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt") or "").strip()
        image_data = str(data.get("image") or "")
        if not prompt or not image_data.startswith("data:image/"):
            return jsonify({"error": "Нужно загрузить изображение и описать изменения."}), 400
        catalog = get_image_models()
        requested = data.get("model", "auto")
        if requested == "auto":
            # Prefer image models that accept reference images, according to their advertised capability descriptors.
            chosen = next((m for m in catalog if "input_references" in (m.get("supported_parameters") or {})), None)
        else:
            chosen = next((m for m in catalog if m.get("id") == requested), None)
        if not chosen:
            return jsonify({"error": "Не найдена модель с поддержкой редактирования по референсу. Выбери другую модель."}), 400
        payload = {"model": chosen["id"], "prompt": prompt, "input_references": [image_data], "aspect_ratio": data.get("aspect_ratio", "auto")}
        result = upstream("POST", "/images", json=payload)
        images = []
        for item in result.get("data", []):
            if item.get("b64_json"):
                media = item.get("media_type") or "image/png"
                images.append("data:" + media + ";base64," + item["b64_json"])
            elif item.get("url"):
                images.append(item["url"])
        usage = result.get("usage") or {}
        return jsonify({"model": chosen["id"], "images": images, "cost": usage.get("cost"), "message": "Готово" if images else "Модель не вернула изображение."})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Файл слишком большой. Максимальный размер запроса — 20 МБ."}), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
