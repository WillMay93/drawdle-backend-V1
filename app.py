# app.py
import os
import json
import re
from datetime import datetime, timezone, date
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from functools import lru_cache

# -------------------- SETUP -------------------- #
app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

CACHE_FILE = "daily_target.json"
LEADERBOARD_FILE = "leaderboard.json"


# -------------------- HELPERS -------------------- #

# Map your UI palette to the names used by the daily target prompt
PALETTE_NAME_BY_HEX = {
    "#000000": "black",
    "#e63946": "red",
    "#457b9d": "blue",
    "#2a9d8f": "green",
    "#f4a261": "orange",
    "#f9c74f": "yellow",
}

def canonical_colour(value: str) -> str:
    """
    Return a canonical colour name in {red,yellow,blue,green,orange,black}.
    Accepts hex or names; defaults to '' if unknown.
    """
    if not value:
        return ""
    v = value.strip().lower()
    if v.startswith("#"):
        # exact palette match (your buttons)
        return PALETTE_NAME_BY_HEX.get(v, "")
    # already a name? keep if in allowed set
    allowed = {"red", "yellow", "blue", "green", "orange", "black"}
    return v if v in allowed else ""


def load_cache():
    """Load cached target if today's is valid."""
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
    """Save today's target to file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(target, f, indent=2)


def clear_leaderboard():
    """Reset leaderboard for new day."""
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
    """Ask OpenAI for a simple drawable target once per day."""
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
    """Basic scoring logic."""
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
    """Expose today's AI-chosen target."""
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
    """Get top 5 leaderboard entries."""
    entries = load_leaderboard()
    entries = sorted(entries, key=lambda e: e.get("score", 0), reverse=True)
    return jsonify(entries[:5])


@app.route("/leaderboard", methods=["POST"])
def post_leaderboard():
    """Save a new leaderboard entry."""
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
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/submit", methods=["POST", "OPTIONS"])
def submit():
    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.get_json(force=True)
        image_base64 = data.get("image_base64")
        attempt = int(data.get("attempt", 1))
        input_colour_raw = str(data.get("colour", ""))  # may be hex or name

        target_info = get_ai_daily_target()
        target_prompt = target_info.get("prompt", "")
        expected_category = target_info.get("category", "")
        expected_colour_raw = target_info.get("colour", "")

        if not image_base64:
            return jsonify({"success": False, "message": "Missing image"}), 400

        # --- Normalise colours ---
        input_colour = canonical_colour(input_colour_raw)
        expected_colour = canonical_colour(expected_colour_raw)

        # image payload
        img_data = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64

        # --- Scoring instruction (nudge it not to parrot the category) ---
        system_instruction = (
            "You are a strict judge for a drawing guessing game. "
            f"TARGET(SECRET): '{target_prompt}'. Expected category: '{expected_category}'. "
            f"Target's main colour name: '{expected_colour}'. "
            "Return ONLY valid JSON with fields: "
            '"score" (0-100), "guess" (short phrase of what the object is), '
            '"correct" (true/false if your guess exactly matches the target word), '
            '"color_match" (true/false if the drawn object’s dominant colour seems to match the target colour), '
            '"shape_match" (true/false if the overall shape resembles the target), '
            '"style_score" (0-25, neatness/clarity), '
            '"category" (the semantic category of YOUR GUESS such as animal, vehicle, fruit, building, tool, etc — '
            "do NOT copy the expected category unless you truly think the drawing is in that category)."
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

        # Extract AI evaluation
        guess = parsed.get("guess", "").strip()
        correct = bool(parsed.get("correct", False))
        ai_color_match = bool(parsed.get("color_match", False))
        shape_match = bool(parsed.get("shape_match", False))
        style_score = int(parsed.get("style_score", 0))
        category = parsed.get("category", "").strip()

        # --- Hard override: if player's brush colour equals expected colour name, count it ---
        color_match = ai_color_match
        if input_colour and expected_colour and input_colour == expected_colour:
            color_match = True

        # Final score with your existing strict scoring
        final_score = strict_score(guess, target_prompt, int(parsed.get("score", 0)), {
            "color_match": color_match,
            "style_score": style_score
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
            "expected_colour": expected_colour,  # normalized name
            "target_id": target_prompt
        })


    except Exception as e:
        print("Error in /submit:", e)
        print("🎨 Received colour from front end:", data.get("colour"))
        print("🎯 Expected colour from target:", target_info.get("colour"))
        print("🎨 Normalised input colour:", canonical_colour(data.get("colour", "")))
        return jsonify({"success": False, "message": str(e)}), 500


# -------------------- MAIN -------------------- #

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
