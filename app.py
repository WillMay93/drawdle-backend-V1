# app.py
import os
import json
import re
import traceback
from datetime import datetime, timezone, date
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from functools import lru_cache

# -------------------- SETUP -------------------- #
app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",             # local dev
    "https://drawdle-frontend.vercel.app"  # deployed frontend
])

@app.after_request
def add_headers(response):
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")
client = OpenAI(api_key=OPENAI_API_KEY)

CACHE_FILE = "daily_target.json"
LEADERBOARD_FILE = "leaderboard.json"


# -------------------- HELPERS -------------------- #

PALETTE_NAME_BY_HEX = {
    "#000000": "black",
    "#e63946": "red",
    "#457b9d": "blue",
    "#2a9d8f": "green",
    "#f4a261": "orange",
    "#f9c74f": "yellow",
}

def canonical_colour(value: str) -> str:
    if not value:
        return ""
    v = value.strip().lower()
    if v.startswith("#"):
        return PALETTE_NAME_BY_HEX.get(v, "")
    allowed = {"red", "yellow", "blue", "green", "orange", "black"}
    return v if v in allowed else ""


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            try:
                data = json.load(f)
                if data.get("date") == date.today().isoformat():
                    return data
            except Exception:
                pass
    return None


def save_cache(target):
    with open(CACHE_FILE, "w") as f:
        json.dump(target, f, indent=2)


def clear_leaderboard():
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump([], f, indent=2)


def load_leaderboard():
    if not os.path.exists(LEADERBOARD_FILE):
        clear_leaderboard()
    with open(LEADERBOARD_FILE, "r") as f:
        return json.load(f)


def save_leaderboard(entries):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(entries, f, indent=2)


@lru_cache(maxsize=1)
def get_ai_daily_target(date_str=None):
    cached = load_cache()
    if cached:
        return cached

    today = date.today().isoformat()
    prompt = (
        "Pick one simple, easily drawable object for a daily sketching game. "
        "It must be something common, recognisable, and fun to draw "
        "(like 'apple', 'car', 'tree', 'dog', 'sun'). "
        "Include a category (fruit, animal, vehicle, etc.) and one main colour "
        "from this list: red, yellow, blue, green, orange, black. "
        "Respond ONLY with valid JSON: "
        '{"prompt":"apple","public_name":"A fruit","category":"fruit","colour":"red"}'
    )

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    text = getattr(response, "output_text", "") or str(response)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        data = {"prompt": "apple", "public_name": "A fruit", "category": "fruit", "colour": "red"}
    else:
        try:
            data = json.loads(m.group(0))
        except Exception:
            data = {"prompt": "apple", "public_name": "A fruit", "category": "fruit", "colour": "red"}

    data["date"] = today
    save_cache(data)
    clear_leaderboard()
    return data


def parse_json_from_text(text):
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def strict_score(guess: str, target: str, ai_score: int, extras: dict):
    if not guess or guess.strip().lower() != target.strip().lower():
        return 0
    score = 50
    if extras.get("color_match"):
        score += 25
    style_score = min(int(extras.get("style_score", 0)), 25)
    score += style_score
    return score


# -------------------- ROUTES -------------------- #

@app.route("/target", methods=["GET"])
def get_target():
    target = get_ai_daily_target()
    return jsonify({
        "date": target["date"],
        "prompt": target["prompt"],
        "public_name": target["public_name"],
        "colour": target["colour"],
        "category": target["category"]
    })


@app.route("/leaderboard", methods=["GET"])
def get_leaderboard():
    entries = load_leaderboard()
    entries = sorted(entries, key=lambda e: e.get("score", 0), reverse=True)
    return jsonify(entries[:5])


@app.route("/leaderboard", methods=["POST"])
def post_leaderboard():
    try:
        data = request.get_json(force=True)
        name = data.get("name", "Unknown")
        score = data.get("score", 0)
        attempts = data.get("attempts", 0)
        image = data.get("image", "")
        timestamp = datetime.now(timezone.utc).isoformat()

        entries = load_leaderboard()
        entries.append({
            "name": name,
            "score": score,
            "attempts": attempts,
            "image": image,
            "timestamp": timestamp
        })

        entries = sorted(entries, key=lambda e: e["score"], reverse=True)[:5]
        save_leaderboard(entries)
        return jsonify({"success": True, "leaderboard": entries})
    except Exception as e:
        print("Error saving leaderboard:", e)
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/submit", methods=["POST", "OPTIONS"])
def submit():
    if request.method == "OPTIONS":
        return "", 200

    try:
        # Safely parse JSON
        try:
            data = request.get_json(force=True)
            if not data:
                raise ValueError("No JSON body received")
        except Exception as e:
            print("❌ Failed to parse JSON:", e)
            return jsonify({"success": False, "message": "Invalid JSON"}), 400

        print("🧩 Received data from frontend:", data)
        print("🧩 Keys:", list(data.keys()))

        image_base64 = data.get("image_base64")
        attempt = int(data.get("attempt", 1))
        input_colour_raw = str(data.get("colour", ""))

        target_info = get_ai_daily_target()
        target_prompt = target_info.get("prompt", "")
        expected_category = target_info.get("category", "")
        expected_colour_raw = target_info.get("colour", "")

        if not image_base64:
            return jsonify({"success": False, "message": "Missing image"}), 400

        input_colour = canonical_colour(input_colour_raw)
        expected_colour = canonical_colour(expected_colour_raw)
        img_data = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64

        system_instruction = (
            "You are a strict judge for a drawing guessing game. "
            f"TARGET(SECRET): '{target_prompt}'. Expected category: '{expected_category}'. "
            f"Target's main colour name: '{expected_colour}'. "
            "Return ONLY valid JSON with fields: "
            '"score" (0-100), "guess", "correct", "color_match", "shape_match", "style_score", "category".'
        )

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_instruction}]},
                {"role": "user", "content": [
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_data}"}
                ]}
            ],
            temperature=0.0
        )

        output_text = getattr(response, "output_text", "") or str(response)
        parsed = parse_json_from_text(output_text) or {}
        if not isinstance(parsed, dict):
            print("⚠️ OpenAI returned unexpected output:", output_text)
            parsed = {}

        guess = parsed.get("guess", "").strip()
        correct = bool(parsed.get("correct", False))
        ai_color_match = bool(parsed.get("color_match", False))
        shape_match = bool(parsed.get("shape_match", False))
        style_score = int(parsed.get("style_score", 0))
        category = parsed.get("category", "").strip()

        color_match = ai_color_match
        if input_colour and expected_colour and input_colour == expected_colour:
            color_match = True

        final_score = strict_score(guess, target_prompt, int(parsed.get("score", 0)), {
            "color_match": color_match,
            "style_score": style_score
        })

        print("✅ Responding with:", {
            "success": correct,
            "score": final_score,
            "guess": guess,
            "category": category,
            "color_match": color_match,
            "shape_match": shape_match,
            "style_score": style_score,
            "expected_category": expected_category,
            "expected_colour": expected_colour,
            "target_id": target_prompt
        })

        return jsonify({
            "success": correct,
            "score": final_score,
            "guess": guess,
            "category": category,
            "color_match": color_match,
            "shape_match": shape_match,
            "style_score": style_score,
            "expected_category": expected_category,
            "expected_colour": expected_colour,
            "target_id": target_prompt
        })

    except Exception as e:
        print("Error in /submit:", e)
        traceback.print_exc()
        print("🎨 Received colour from front end:", data.get("colour") if 'data' in locals() else "N/A")
        print("🎯 Expected colour from target:", target_info.get("colour") if 'target_info' in locals() else "N/A")
        return jsonify({"success": False, "message": str(e)}), 500


# -------------------- MAIN -------------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
