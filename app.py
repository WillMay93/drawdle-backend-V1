# app.py
import os
import json
import re
import traceback
import hashlib
from datetime import datetime, timezone, date

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# -------------------- SETUP -------------------- #
app = Flask(__name__)

# CORS restricted to your known frontends (same as OLD)
CORS(app, origins=[
    "http://localhost:3000",               # local dev
    "https://drawdle-frontend.vercel.app"  # deployed frontend
])

@app.after_request
def add_headers(response):
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

# OpenAI API key from environment (same pattern as OLD)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)

CACHE_FILE = "daily_target.json"
LEADERBOARD_FILE = "leaderboard.json"

# -------------------- HELPERS -------------------- #

# Map hex colors to canonical names
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
        return PALETTE_NAME_BY_HEX.get(v, "")
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


def get_ai_daily_target():
    """
    Ask OpenAI for a simple drawable target once per day.
    Uses a deterministic seed based on today's date to avoid repeats.
    """
    today = date.today().isoformat()

    cached = load_cache()
    if cached and cached.get("date") == today:
        return cached

    # deterministic seed for variety
    date_hash = int(hashlib.md5(today.encode()).hexdigest(), 16)
    seed_number = date_hash % 1000

    prompt = (
        f"You are creating a daily challenge for a drawing game. Today's seed number is {seed_number}. "
        "Pick ONE simple, easily drawable object that is common and recognizable. "
        "Choose from diverse categories like: fruits, vegetables, animals, vehicles, buildings, "
        "nature items, household objects, tools, sports equipment, or food items. "
        "Avoid repeating overly common choices like 'apple'. "
        "Pick one main colour from: red, yellow, blue, green, orange, black. "

        "For the 'location' field, provide a highly specific, realistic place relevant to the object: "
        "- If it's a fruit/vegetable: where it's grown (e.g., 'grown in Spain', 'native to Thailand'). "
        "- If it's a household item: where it is stored (e.g., 'kept under the kitchen sink'). "
        "- If it's an animal: its natural habitat or home region. "
        "- If it's a tool: where it is commonly found (e.g., 'garage workshop'). "

        "Respond ONLY with valid JSON in this exact format: "
        '{"prompt":"YOUR_OBJECT","public_name":"A category hint","category":"category_name","colour":"colour_name","location":"exact_location"}'
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.2
    )

    text = response.choices[0].message.content
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        data = {
            "prompt": "apple",
            "public_name": "A fruit",
            "category": "fruit",
            "colour": "red",
            "location": "kitchen"
        }
    else:
        try:
            data = json.loads(m.group(0))
            if "location" not in data:
                data["location"] = "unknown"
        except Exception:
            data = {
                "prompt": "apple",
                "public_name": "A fruit",
                "category": "fruit",
                "colour": "red",
                "location": "kitchen"
            }

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


def generate_hint(guess: str, target: str, category: str, expected_colour: str, attempt: int):
    """Generate a contextual hint based on the player's guess and progress."""
    if attempt == 1:
        return f"Try drawing something in the {category} category!"
    elif attempt == 2:
        return f"Hint: The target colour is {expected_colour}."
    elif attempt == 3:
        if guess.lower() != target.lower():
            return f"You drew '{guess}', but we're looking for something else in the {category} category."
        else:
            return "You're very close! Check the colour and shape details."
    else:
        return f"Almost there! Remember: {category}, {expected_colour} colour. Focus on key features!"


def normalize_category(cat: str) -> str:
    """Normalize category for comparison."""
    if not cat:
        return ""
    cat = cat.strip().lower()
    # Remove common plural 's'
    if cat.endswith("s") and len(cat) > 1:
        cat = cat[:-1]
    return cat


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
        "category": target["category"],
        "location": target.get("location", "unknown")
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
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/submit", methods=["POST", "OPTIONS"])
def submit():
    if request.method == "OPTIONS":
        return "", 200

    try:
        # Safer JSON parsing with debug (from OLD)
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
        target_prompt = target_info.get("prompt", "").strip().lower()
        expected_category = target_info.get("category", "").strip().lower()
        expected_colour_raw = target_info.get("colour", "")
        target_location = target_info.get("location", "unknown")

        if not image_base64:
            return jsonify({"success": False, "message": "Missing image"}), 400

        # Normalize colours
        input_colour = canonical_colour(input_colour_raw)
        expected_colour = canonical_colour(expected_colour_raw)

        # Image payload
        img_data = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64

        # Step 1: Get AI's independent guess
        guess_instruction = (
            "You are analyzing a drawing. Look at this image and tell me:\n"
            "1. What object is drawn in this image? (be specific)\n"
            "2. What category does it belong to? (e.g., fruit, animal, vehicle, plant, tool, food, nature item, etc.)\n"
            "3. What is the primary color used in the drawing?\n"
            "4. How neat and clear is the drawing? (0-25 points for style/neatness)\n\n"
            "Return ONLY valid JSON with these exact fields:\n"
            '{"guess": "what_object_is_drawn", '
            '"category": "category_name", '
            '"primary_color": "color_name", '
            '"style_score": 0-25}'
        )

        # Get independent guess from AI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": guess_instruction},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
                ]}
            ],
            temperature=0.3
        )

        output_text = response.choices[0].message.content
        parsed = parse_json_from_text(output_text) or {}

        # Extract AI's independent evaluation
        guess = parsed.get("guess", "").strip().lower()
        ai_category = parsed.get("category", "").strip().lower()
        ai_detected_color = parsed.get("primary_color", "").strip().lower()
        style_score = max(0, min(25, int(parsed.get("style_score", 0))))

        # Step 2: Compare against target
        # Check if guess matches target (exact or very close)
        correct = guess == target_prompt

        if not correct:
            guess_normalized = guess.replace(" ", "").replace("-", "")
            target_normalized = target_prompt.replace(" ", "").replace("-", "")

            # Allow singular/plural variations
            if guess_normalized.endswith("s"):
                guess_normalized = guess_normalized[:-1]
            if target_normalized.endswith("s"):
                target_normalized = target_normalized[:-1]

            correct = guess_normalized == target_normalized

        # Step 3: Evaluate shape match if we need more detail
        shape_match = False
        if correct:
            shape_instruction = (
                f"Look at this drawing. The person was trying to draw a '{target_prompt}'. "
                f"Does the shape and form clearly represent a {target_prompt}? "
                f"Return ONLY valid JSON: "
                f'{{"shape_match": true/false}}'
            )

            shape_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": shape_instruction},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
                    ]}
                ],
                temperature=0.2
            )

            shape_parsed = parse_json_from_text(shape_response.choices[0].message.content) or {}
            shape_match = bool(shape_parsed.get("shape_match", False))

        # Color match evaluation (from AI)
        ai_color_match = canonical_colour(ai_detected_color) == expected_colour

        # Normalize categories
        normalized_ai_category = normalize_category(ai_category)
        normalized_expected_category = normalize_category(expected_category)

        # Expanded category match logic with synonyms
        category_synonyms = {
            "nature item": ["plant", "flower", "tree", "nature", "vegetation"],
            "plant": ["nature item", "vegetation", "flower", "tree"],
            "fruit": ["food", "produce"],
            "food": ["fruit", "vegetable", "produce", "snack"],
            "vehicle": ["transport", "car", "transportation"],
            "animal": ["creature", "pet", "wildlife"],
            "tool": ["utensil", "instrument", "equipment"],
        }

        # Direct or containment match
        category_match = (
            normalized_ai_category == normalized_expected_category or
            (normalized_ai_category and normalized_ai_category in normalized_expected_category) or
            (normalized_expected_category and normalized_expected_category in normalized_ai_category)
        )

        # Check synonyms if direct match fails
        if not category_match:
            expected_synonyms = category_synonyms.get(normalized_expected_category, [])
            ai_synonyms = category_synonyms.get(normalized_ai_category, [])

            category_match = (
                normalized_ai_category in expected_synonyms or
                normalized_expected_category in ai_synonyms
            )

        # Color match - if player used the expected color, override AI judgement
        color_match = ai_color_match
        if input_colour and expected_colour and input_colour == expected_colour:
            color_match = True

        # Generate contextual hint & location hint
        hint = generate_hint(guess, target_prompt, expected_category, expected_colour, attempt)
        hint_location = target_location

        # Success determination
        success = correct and category_match

        # Calculate score
        if success:
            base_score = 50
            if color_match:
                base_score += 20
            if shape_match:
                base_score += 20
            base_score += style_score  # 0-25 points
            attempt_penalty = (attempt - 1) * 10
            final_score = max(0, min(100, base_score - attempt_penalty))
        else:
            final_score = 0

        print(f"\n=== DEBUG ===")
        print(f"Target: {target_prompt}")
        print(f"AI Guess: {guess}")
        print(f"Correct match: {correct}")
        print(f"Expected category: {expected_category} (normalized: {normalized_expected_category})")
        print(f"AI category: {ai_category} (normalized: {normalized_ai_category})")
        print(f"Category match: {category_match}")
        print(f"Expected colour: {expected_colour}")
        print(f"AI detected colour: {ai_detected_color}")
        print(f"Input colour (brush): {input_colour}")
        print(f"Color match: {color_match}")
        print(f"Shape match: {shape_match}")
        print(f"Success: {success}")
        print(f"=============\n")

        return jsonify({
            "success": success,
            "score": final_score,
            "guess": guess,
            "category": ai_category,
            "category_match": category_match,
            "color_match": color_match,
            "shape_match": shape_match,
            "style_score": style_score,
            "expected_category": expected_category,
            "expected_colour": expected_colour,
            "target_id": target_prompt,
            "hint": hint,
            "hint_location": hint_location
        })

    except Exception as e:
        print("Error in /submit:", e)
        traceback.print_exc()
        print("🎨 Received colour from front end:", data.get("colour") if 'data' in locals() else "N/A")
        print("🎯 Expected colour from target:", target_info.get("colour") if 'target_info' in locals() else "N/A")
        return jsonify({"success": False, "message": str(e)}), 500


# -------------------- MAIN -------------------- #

if __name__ == "__main__":
    # Deployment-friendly: use PORT from env, host 0.0.0.0, debug off by default
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
