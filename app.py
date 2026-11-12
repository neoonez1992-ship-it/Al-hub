import os
import time
import requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

BASE = "https://openrouter.ai/api/v1"
_models_cache = {"at": 0, "data": []}
MODELS_CACHE_SECONDS = 300


def api_key():
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def auth_headers():
    key = api_key()
    if not key:
        raise RuntimeError("На сервере не задан OPENROUTER_API_KEY. Добавь его в Environment на Render.")
    return {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("APP_URL", "https://al-hub.onrender.com"),
        "X-Title": "AI HUB",
    }


def get_all_models(force=False):
    now = time.time()
    if not force and _models_cache["data"] and now - _models_cache["at"] < MODELS_CACHE_SECONDS:
        return _models_cache["data"]

    response = requests.get(BASE + "/models", timeout=45)
    response.raise_for_status()
    raw = response.json().get("data", [])
    models = []

    for item in raw:
        pricing = item.get("pricing") or {}
        architecture = item.get("architecture") or {}
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")

        try:
            is_free = float(prompt_price or 0) == 0 and float(completion_price or 0) == 0
        except (TypeError, ValueError):
            is_free = False

        models.append({
            "id": item.get("id", ""),
            "name": item.get("name") or item.get("id", "Без названия"),
            "description": item.get("description") or "",
            "context_length": item.get("context_length"),
            "pricing": {
                "prompt": prompt_price,
                "completion": completion_price,
                "image": pricing.get("image"),
                "request": pricing.get("request"),
            },
            "free": is_free,
            "input_modalities": architecture.get("input_modalities") or ["text"],
            "output_modalities": architecture.get("output_modalities") or ["text"],
            "supported_parameters": item.get("supported_parameters") or [],
            "created": item.get("created"),
        })

    models.sort(key=lambda x: (not x["free"], (x["name"] or "").lower()))
    _models_cache["at"] = now
    _models_cache["data"] = models
    return models


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        ok=True,
        key_configured=bool(api_key()),
        provider="OpenRouter",
    )


@app.get("/api/models")
def models():
    try:
        force = request.args.get("refresh") == "1"
        all_models = get_all_models(force=force)
        return jsonify(models=all_models, total=len(all_models), updated_at=int(_models_cache["at"]))
    except requests.RequestException:
        return jsonify(error="Не удалось получить каталог OpenRouter. Попробуй обновить список позже."), 502
    except Exception:
        app.logger.exception("Models endpoint failed")
        return jsonify(error="Ошибка при загрузке каталога моделей."), 500


def ask(model, messages, max_tokens=1200):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        BASE + "/chat/completions",
        headers=auth_headers(),
        json=payload,
        timeout=(20, 150),
    )

    if not response.ok:
        try:
            data = response.json()
            error = data.get("error") or {}
            message = error.get("message") or response.text
            code = error.get("code")
            if code:
                message = f"{message} (код {code})"
        except (ValueError, AttributeError):
            message = response.text or f"HTTP {response.status_code}"
        raise RuntimeError(str(message)[:700])

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Модель не вернула ответ.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")

    # Некоторые модели возвращают мультимодальный ответ массивом блоков.
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    parts.append("[Модель вернула изображение]")
        content = "\n".join(x for x in parts if x)
    if not content and message.get("refusal"):
        content = message["refusal"]
    return str(content or "Модель вернула пустой ответ.")


@app.post("/api/chat")
def chat():
    try:
        body = request.get_json(force=True) or {}
        model = str(body.get("model", "")).strip()
        prompt = str(body.get("prompt", "")).strip()
        image_data = body.get("image")

        if not model or (not prompt and not image_data):
            return jsonify(error="Выбери модель и напиши запрос или прикрепи изображение."), 400

        history = body.get("history") or []
        messages = []
        for item in history[-12:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({"role": role, "content": content[:12000]})

        user_content = prompt[:12000]
        if image_data:
            if not isinstance(image_data, str) or not image_data.startswith("data:image/"):
                return jsonify(error="Не удалось прочитать изображение. Прикрепи JPG, PNG или WebP."), 400
            if len(image_data) > 8 * 1024 * 1024:
                return jsonify(error="Изображение слишком большое. Выбери файл меньшего размера."), 400
            content = []
            if user_content:
                content.append({"type": "text", "text": user_content})
            content.append({
                "type": "image_url",
                "image_url": {"url": image_data}
            })
            user_content = content

        messages.append({"role": "user", "content": user_content})
        answer = ask(model, messages)
        # Не сохраняем image data URL в историю браузера/ответе.
        return jsonify(answer=answer)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 400
    except requests.Timeout:
        return jsonify(error="Истекло время ожидания ответа модели. Попробуй ещё раз или выбери другую модель."), 504
    except requests.RequestException:
        return jsonify(error="Не удалось связаться с OpenRouter. Проверь соединение и попробуй ещё раз."), 502
    except Exception:
        app.logger.exception("Chat endpoint failed")
        return jsonify(error="Внутренняя ошибка сервера. Подробности доступны в логах Render."), 500


@app.post("/api/compare")
def compare():
    try:
        body = request.get_json(force=True) or {}
        prompt = str(body.get("prompt", "")).strip()
        chosen = [str(x).strip() for x in (body.get("models") or [])[:3] if str(x).strip()]
        if not prompt or not chosen:
            return jsonify(error="Напиши запрос и выбери хотя бы одну модель."), 400

        results = []
        for model in chosen:
            try:
                answer = ask(model, [{"role": "user", "content": prompt[:10000]}], max_tokens=1000)
                results.append({"model": model, "answer": answer})
            except Exception as exc:
                results.append({"model": model, "error": str(exc)[:500]})
        return jsonify(results=results)
    except Exception:
        app.logger.exception("Compare endpoint failed")
        return jsonify(error="Ошибка сравнения. Подробности доступны в логах Render."), 500


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Запрос слишком большой. Уменьши размер изображения."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8765)))
