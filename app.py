
import os
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

cache = {"models": None, "images": None}


def headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": SITE_URL or "https://ai-hub.local",
        "X-Title": SITE_NAME,
        "Content-Type": "application/json"
    }


def require_key():
    if not API_KEY:
        return jsonify({
            "error": "Не задан OPENROUTER_API_KEY в настройках Render."
        }), 500
    return None


def upstream(method, path, **kwargs):
    r = requests.request(
        method,
        BASE + path,
        headers=headers(),
        timeout=TIMEOUT,
        **kwargs
    )
    try:
        data = r.json()
    except Exception:
        data = {"error": r.text[:1000]}

    if not r.ok:
        error = data.get("error", data)
        if isinstance(error, dict):
            error = error.get("message", str(error))
        raise RuntimeError(str(error))

    return data


def get_models():
    if cache["models"] is None:
        cache["models"] = upstream("GET", "/models").get("data", [])
    return cache["models"]


def get_image_models():
    if cache["images"] is None:
        cache["images"] = upstream("GET", "/images/models").get("data", [])
    return cache["images"]


def is_vision(model):
    modalities = (
        model.get("architecture") or {}
    ).get("input_modalities", [])
    return "image" in modalities


def describe_model(model):
    text = " ".join([
        str(model.get("id", "")),
        str(model.get("name", "")),
        str(model.get("description", ""))
    ]).lower()

    descriptions = []

    if is_vision(model):
        descriptions.append("Анализ изображений и фотографий")

    if re.search(r"coder|codestral|code", text):
        descriptions.append("Программирование и написание кода")

    if re.search(r"reason|thinking|qwq|deepseek-r1", text):
        descriptions.append("Логика и сложные рассуждения")

    if not descriptions:
        descriptions.append("Общение, тексты и универсальные задачи")

    return ". ".join(descriptions)


def choose_model(prompt, has_images=False, requested="auto"):
    catalog = get_models()

    if requested and requested != "auto":
        model = next(
            (m for m in catalog if m.get("id") == requested),
            None
        )
        if not model:
            raise RuntimeError("Модель отсутствует в каталоге.")
        if has_images and not is_vision(model):
            raise RuntimeError(
                "Выбранная модель не поддерживает изображения."
            )
        return model

    text = (prompt or "").lower()

    coding = bool(re.search(
        r"код|программ|python|javascript|html|css|"
        r"ошибк|сайт|приложен|скрипт|разработ",
        text
    ))

    reasoning = bool(re.search(
        r"проанализ|почему|сравни|докажи|"
        r"математ|логик|рассчитай|стратег",
        text
    ))

    candidates = [
        m for m in catalog
        if not has_images or is_vision(m)
    ]

    if not candidates:
        raise RuntimeError(
            "Нет доступных моделей с поддержкой изображений."
        )

    def score(model):
        name = (
            model.get("id", "") + " " +
            model.get("name", "")
        ).lower()

        points = 0

        if coding and re.search(r"coder|codestral|code|deepseek", name):
            points += 8

        if reasoning and re.search(
            r"reason|thinking|r1|qwq|opus|sonnet", name
        ):
            points += 7

        if not coding and not reasoning and re.search(
            r"gpt|gemini|claude|qwen|llama", name
        ):
            points += 4

        if has_images and is_vision(model):
            points += 10

        pricing = model.get("pricing") or {}

        try:
            if (
                float(pricing.get("prompt", 1)) == 0
                and float(pricing.get("completion", 1)) == 0
            ):
                points += 1.5
        except (ValueError, TypeError):
            pass

        if model.get("context_length", 0) >= 32000:
            points += 0.5

        return points

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def get_answer(result):
    choices = result.get("choices") or []

    if not choices:
        return result.get("output_text", "")

    content = (
        choices[0].get("message") or {}
    ).get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )

    return str(content)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "configured": bool(API_KEY),
        "service": "AI HUB"
    })


@app.get("/api/models")
def models():
    error = require_key()
    if error:
        return error

    try:
        catalog = get_models()
        result = []

        for m in catalog:
            pricing = m.get("pricing") or {}

            result.append({
                "id": m.get("id"),
                "name": m.get("name") or m.get("id"),
                "description": describe_model(m),
                "context_length": m.get("context_length"),
                "vision": is_vision(m),
                "pricing": pricing,
                "free": (
                    pricing.get("prompt") == "0"
                    and pricing.get("completion") == "0"
                )
            })

        # Keep compatibility with the original API.
        return jsonify({
            "models": result,
            "data": result,
            "total": len(result)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.get("/api/images/models")
def image_models():
    error = require_key()
    if error:
        return error

    try:
        return jsonify({"data": get_image_models()})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.post("/api/chat")
def chat():
    error = require_key()
    if error:
        return error

    try:
        data = request.get_json(force=True) or {}

        prompt = str(data.get("prompt") or "").strip()
        image = data.get("image")
        mode = data.get("mode", "auto")

        # Support both the new frontend and the old API.
        requested = (
            data.get("model", "auto")
            if mode == "manual"
            else "auto"
        )

        messages = data.get("history") or data.get("messages") or []

        if not messages:
            messages = []

        messages = [
            m for m in messages
            if isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
        ][-20:]

        has_images = bool(image)

        if image and not str(image).startswith("data:image/"):
            return jsonify({
                "error": "Некорректный формат изображения."
            }), 400

        if image:
            content = [
                {"type": "text", "text": prompt or "Проанализируй изображение."},
                {
                    "type": "image_url",
                    "image_url": {"url": image}
                }
            ]
            messages.append({
                "role": "user",
                "content": content
            })

        elif prompt:
            messages.append({
                "role": "user",
                "content": prompt
            })

        else:
            return jsonify({"error": "Напиши запрос."}), 400

        chosen = choose_model(
            prompt,
            has_images,
            requested
        )

        payload = {
            "model": chosen["id"],
            "messages": messages,
            "max_tokens": 3000
        }

        result = upstream(
            "POST",
            "/chat/completions",
            json=payload
        )

        answer = get_answer(result)

        return jsonify({
            "answer": answer,
            "reply": answer,
            "selected_model": chosen["id"],
            "model": chosen["id"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.post("/api/compare")
def compare():
    error = require_key()
    if error:
        return error

    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt") or "").strip()
        ids = data.get("models") or []

        if not prompt or not ids:
            return jsonify({
                "error": "Напиши запрос и выбери модели."
            }), 400

        catalog = get_models()
        available = {m.get("id") for m in catalog}

        results = []

        for model_id in ids[:3]:
            if model_id not in available:
                results.append({
                    "model": model_id,
                    "error": "Модель отсутствует в каталоге."
                })
                continue

            try:
                result = upstream(
                    "POST",
                    "/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 2000
                    }
                )

                answer = get_answer(result)

                results.append({
                    "model": model_id,
                    "answer": answer,
                    "reply": answer
                })

            except Exception as e:
                results.append({
                    "model": model_id,
                    "error": str(e)
                })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.post("/api/images/generate")
def generate_image():
    error = require_key()
    if error:
        return error

    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt") or "").strip()

        if not prompt:
            return jsonify({"error": "Напиши описание изображения."}), 400

        catalog = get_image_models()
        requested = data.get("model", "auto")

        if requested == "auto":
            chosen = catalog[0] if catalog else None
        else:
            chosen = next(
                (m for m in catalog if m.get("id") == requested),
                None
            )

        if not chosen:
            return jsonify({
                "error": "Модель генерации не найдена."
            }), 400

        result = upstream(
            "POST",
            "/images",
            json={
                "model": chosen["id"],
                "prompt": prompt,
                "aspect_ratio": data.get("aspect_ratio", "1:1"),
                "resolution": data.get("resolution", "1K")
            }
        )

        images = []

        for item in result.get("data", []):
            if item.get("url"):
                images.append(item["url"])
            elif item.get("b64_json"):
                media = item.get("media_type") or "image/png"
                images.append(
                    f"data:{media};base64,{item['b64_json']}"
                )

        return jsonify({
            "model": chosen["id"],
            "images": images,
            "message": "Готово" if images else "Изображение не получено."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.post("/api/images/edit")
def edit_image():
    error = require_key()
    if error:
        return error

    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt") or "").strip()
        image = str(data.get("image") or "")

        if not prompt or not image.startswith("data:image/"):
            return jsonify({
                "error": "Загрузи изображение и опиши изменения."
            }), 400

        catalog = get_image_models()
        requested = data.get("model", "auto")

        if requested == "auto":
            chosen = next(
                (
                    m for m in catalog
                    if "input_references" in (
                        m.get("supported_parameters") or []
                    )
                ),
                None
            )
        else:
            chosen = next(
                (m for m in catalog if m.get("id") == requested),
                None
            )

        if not chosen:
            return jsonify({
                "error": "Модель редактирования не найдена."
            }), 400

        result = upstream(
            "POST",
            "/images",
            json={
                "model": chosen["id"],
                "prompt": prompt,
                "input_references": [image],
                "aspect_ratio": data.get("aspect_ratio", "auto")
            }
        )

        images = []

        for item in result.get("data", []):
            if item.get("url"):
                images.append(item["url"])
            elif item.get("b64_json"):
                media = item.get("media_type") or "image/png"
                images.append(
                    f"data:{media};base64,{item['b64_json']}"
                )

        return jsonify({
            "model": chosen["id"],
            "images": images,
            "message": "Готово" if images else "Изображение не получено."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "error": "Файл слишком большой. Максимум 20 МБ."
    }), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000"))
    )
