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
    """Ask OpenAI for a simple drawable target once per day with category variety."""
    today = date.today().isoformat()

    cached = load_cache()
    if cached and cached.get("date") == today:
        return cached

    # Load previous target to avoid category repetition
    previous_category = None
    if cached:
        previous_category = cached.get("category", "").strip().lower()

    import hashlib
    date_hash = int(hashlib.md5(today.encode()).hexdigest(), 16)
    seed_number = date_hash % 1000

    # Build category exclusion text
    avoid_text = ""
    if previous_category:
        avoid_text = f"IMPORTANT: Yesterday's category was '{previous_category}'. You MUST choose a DIFFERENT category today. "

    prompt = (
        f"You are creating a daily challenge for a drawing game. Today's seed number is {seed_number}. "
        f"{avoid_text}"
        "Pick ONE simple, easily drawable object that is common and recognizable. "
        
        "Choose from these DIVERSE categories (pick a different one each day): "
        "- Musical instruments (guitar, piano, drum, trumpet, violin, etc.)\n"
        "- Clothing (hat, shoe, shirt, dress, glove, etc.)\n"
        "- Sea creatures (fish, octopus, crab, dolphin, whale, etc.)\n"
        "- Fruits (apple, banana, orange, strawberry, watermelon, pear, etc.)\n"
        "- Vegetables (carrot, broccoli, tomato, corn, pepper, etc.)\n"
        "- Animals (cat, dog, bird, fish, elephant, butterfly, etc.)\n"
        "- Vehicles (car, bicycle, boat, airplane, train, etc.)\n"
        "- Buildings (house, castle, lighthouse, barn, skyscraper, etc.)\n"
        "- Nature items (tree, flower, sun, cloud, mountain, star, etc.)\n"
        "- Household objects (chair, lamp, cup, door, window, bed, etc.)\n"
        "- Tools (hammer, wrench, saw, screwdriver, paintbrush, etc.)\n"
        "- Sports equipment (ball, tennis racket, skateboard, hockey stick, etc.)\n"
        "- Food items (pizza, cake, sandwich, ice cream, burger, etc.)\n"
        "- Weather elements (rainbow, lightning, snowflake, raindrop, etc.)\n"
        
        "Pick one main colour from: red, yellow, blue, green, orange, black. "

        "For the 'location' field, provide a highly specific, realistic place relevant to the object: "
        "- If it's a fruit/vegetable: where it's grown (e.g., 'grown in orchards in Washington state'). "
        "- If it's a household item: where it is stored (e.g., 'kept in the bathroom cabinet'). "
        "- If it's an animal: its natural habitat (e.g., 'found in African savannas'). "
        "- If it's a tool: where it is commonly found (e.g., 'stored in garage toolboxes'). "
        "- If it's a vehicle: where it operates (e.g., 'seen on highways', 'found at marinas'). "
        "- If it's food: where it's commonly served (e.g., 'served at pizzerias', 'sold at bakeries'). "

        "Respond ONLY with valid JSON in this exact format: "
        '{"prompt":"YOUR_OBJECT","public_name":"A category hint","category":"category_name","colour":"colour_name","location":"exact_location","use_for":"what_the_object_is_used_for"}'
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.3  # Slightly higher temperature for more variety
    )

    text = response.choices[0].message.content
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        # Fallback with variety
        fallbacks = [
            {
                "prompt": "bicycle",
                "public_name": "A vehicle",
                "category": "vehicle",
                "colour": "blue",
                "location": "parked on city streets"
            },
            {
                "prompt": "guitar",
                "public_name": "A musical instrument",
                "category": "musical instrument",
                "colour": "orange",
                "location": "kept in music studios"
            },
            {
                "prompt": "lighthouse",
                "public_name": "A building",
                "category": "building",
                "colour": "red",
                "location": "standing on rocky coastlines"
            }
        ]
        data = fallbacks[seed_number % len(fallbacks)]
    else:
        try:
            data = json.loads(m.group(0))
            # Ensure location exists
            if "location" not in data:
                data["location"] = "unknown"
            
            # Double-check we didn't repeat category
            if previous_category and data.get("category", "").strip().lower() == previous_category:
                print(f"WARNING: AI repeated category '{previous_category}', using fallback")
                # Use a different fallback based on seed
                fallbacks = [
                    {"prompt": "hammer", "category": "tool", "colour": "black", "location": "garage workshops"},
                    {"prompt": "dolphin", "category": "sea creature", "colour": "blue", "location": "tropical oceans"},
                    {"prompt": "rainbow", "category": "weather element", "colour": "red", "location": "seen in the sky after rain"}
                ]
                fb = fallbacks[seed_number % len(fallbacks)]
                data.update(fb)
                if "public_name" not in data:
                    data["public_name"] = f"A {fb['category']}"
                    
        except Exception as e:
            print(f"Error parsing AI response: {e}")
            data = {
                "prompt": "drum",
                "public_name": "A musical instrument",
                "category": "musical instrument",
                "colour": "red",
                "location": "played in concert halls"
            }

    data["date"] = today
    save_cache(data)
    clear_leaderboard()
    
    print(f"Generated target for {today}: {data['prompt']} (category: {data['category']})")
    if previous_category:
        print(f"Previous category was: {previous_category}")
    
    return data


def parse_json_from_text(text):
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def generate_hint(guess: str, target: str, category: str, expected_colour: str, location: str, attempt: int):
    """
    Generate a contextual hint based on the player's guess, progress,
    AND the target location.
    """
    # Capitalize category for better readability
    category_display = category.title() if category else "this category"
    colour_text = expected_colour or "a specific colour"
    location_text = location or "its usual place"
    
    if attempt == 1:
        # Very gentle nudge
        return f"Try drawing something in the {category_display} category."
    elif attempt == 2:
        # Add colour clue
        return f"Hint: This object is usually {colour_text}."
    elif attempt == 3:
        # Bring in location
        return f"Hint: It's typically {location_text}"
    elif attempt == 4:
        # Strong hint: category + colour + location
        return (
            f"Almost there! It's in the {category_display} category, "
            f"usually {colour_text}, and you'd typically find it {location_text}"
        )
    else:
        # Attempt 5 - very strong hint
        return (
            f"Last chance! Think of a {colour_text} {category_display.lower()} "
            f"that is {location_text}"
        )


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
        "location": target.get("location", "unknown"),
         "use_for": target.get("use_for", "unknown")
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
        attempt = int(data.get("attempt", 1) or 1)
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

        # -------------------- STEP 1: AI independent analysis -------------------- #
        guess_instruction = (
            "You are analyzing a player's drawing from a browser paint app for a guessing game.\n"
            "Look at the image and provide:\n"
            "1. 'guess' – what single main object is drawn (be specific, 1–3 words).\n"
            "2. 'category' – high-level category (fruit, animal, vehicle, plant, tool, food, nature item, etc.).\n"
            "3. 'primary_color' – main colour of the object.\n"
            "4. 'style_score' – 0–25, based on neatness, clarity, line confidence, and overall polish.\n"
            "5. 'background_score' – 0–20, where 0 means no background (just object on blank space), "
            "and 20 means a rich background or scene (sky, ground, room, scenery, multiple environment elements).\n"
            "6. 'creativity_score' – 0–30, based on extra details, composition, expression, shading, fun ideas, "
            "and overall creativity.\n\n"
            "Return ONLY valid JSON with these exact fields:\n"
            '{'
            '"guess":"what_object_is_drawn",'
            '"category":"category_name",'
            '"primary_color":"color_name",'
            '"style_score":0-25,'
            '"background_score":0-20,'
            '"creativity_score":0-30'
            '}'
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini-vision",
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

        # Raw component scores from AI (clamped)
        style_score_raw = int(parsed.get("style_score", 0) or 0)
        background_score_raw = int(parsed.get("background_score", 0) or 0)
        creativity_score_raw = int(parsed.get("creativity_score", 0) or 0)

        style_score_raw = max(0, min(style_score_raw, 25))
        background_score_raw = max(0, min(background_score_raw, 20))
        creativity_score_raw = max(0, min(creativity_score_raw, 30))

        # -------------------- STEP 2: Compare against target -------------------- #

        # Check if guess matches target (exact match or close variant)
        correct = guess == target_prompt

        if not correct:
            guess_normalized = guess.replace(" ", "").replace("-", "")
            target_normalized = target_prompt.replace(" ", "").replace("-", "")

            if guess_normalized.endswith("s"):
                guess_normalized = guess_normalized[:-1]
            if target_normalized.endswith("s"):
                target_normalized = target_normalized[:-1]

            correct = guess_normalized == target_normalized

        # Shape match evaluation (extra strictness but not instant fail)
        shape_match = False
        if correct:
            shape_instruction = (
                f"Look at this drawing. The person was trying to draw a '{target_prompt}'. "
                f"Does the shape and form clearly represent a {target_prompt}? "
                f"Return ONLY valid JSON: {{\"shape_match\": true/false}}"
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

        # Color match evaluation
        ai_color_match = canonical_colour(ai_detected_color) == expected_colour

        # Normalize categories for comparison
        normalized_ai_category = normalize_category(ai_category)
        normalized_expected_category = normalize_category(expected_category)

        # Expanded category match logic with synonyms
        category_synonyms = {
        }

        # Category match logic
        category_match = (
            normalized_ai_category == normalized_expected_category or
            (normalized_ai_category and normalized_ai_category in normalized_expected_category) or
            (normalized_expected_category and normalized_expected_category in normalized_ai_category)
        )

        if not category_match:
            expected_synonyms = category_synonyms.get(normalized_expected_category, [])
            ai_synonyms = category_synonyms.get(normalized_ai_category, [])

            category_match = (
                normalized_ai_category in expected_synonyms or
                normalized_expected_category in ai_synonyms
            )

        # Color match - if player used the expected colour, override AI
        color_match = ai_color_match
        if input_colour and expected_colour and input_colour == expected_colour:
            color_match = True

        # -------------------- STEP 3: Scoring (stricter, with style/background/creativity) -------------------- #

        # SUCCESS is about the game outcome (for your UI):
        success = bool(correct and category_match)

        # Weighting:
        # - Object correctness: up to 40
        # - Shape: up to 15
        # - Colour: up to 10
        # - Style: up to 30 (strong weight)
        # - Background: up to 10
        # - Creativity: up to 15
        # Then attempt penalty (10 per extra attempt), all clamped to 0–100.

        # Scale the AI raw scores into point values
        style_points = min(30, int(round(style_score_raw * 1.2)))              # 0–30
        background_points = int(round(background_score_raw * 0.5))            # 0–10
        creativity_points = int(round(creativity_score_raw * 0.5))            # 0–15

        # Core correctness-dependent components
        object_points = 40 if success else 0
        shape_points = 15 if (success and shape_match) else 0
        colour_points = 10 if (success and color_match) else 0

        # Stricter but not brutal:
        # - If success: full scoring with attempt penalty
        # - If failure: you can still get some style/background/creativity points, but capped lower
        attempt = max(1, attempt)
        attempt_penalty = (attempt - 1) * 10

        if success:
            raw_score = (
                object_points +
                shape_points +
                colour_points +
                style_points +
                background_points +
                creativity_points
            )
            final_score = max(0, min(100, raw_score - attempt_penalty))
        else:
            # Wrong object -> only “style” side can score, and not too high
            raw_style_side = style_points + background_points + creativity_points
            final_score = max(0, min(60, int(round(raw_style_side * 0.6))))

        # -------------------- STEP 4: Hint & debug -------------------- #

        hint = generate_hint(
            guess=guess,
            target=target_prompt,
            category=expected_category,
            expected_colour=expected_colour,
            location=target_location,
            attempt=attempt
        )
        hint_location = target_location if attempt >= 3 else ""

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
        print(f"Style score raw: {style_score_raw} -> {style_points}")
        print(f"Background score raw: {background_score_raw} -> {background_points}")
        print(f"Creativity score raw: {creativity_score_raw} -> {creativity_points}")
        print(f"Attempt: {attempt}, penalty: {attempt_penalty}")
        print(f"Success: {success}, Final score: {final_score}")
        print(f"=============\n")

        # -------------------- STEP 5: Progress-bar friendly breakdown -------------------- #

        breakdown = {
            "object": {"score": object_points, "max": 40},
            "shape": {"score": shape_points, "max": 15},
            "colour": {"score": colour_points, "max": 10},
            "style": {"score": style_points, "max": 30},
            "background": {"score": background_points, "max": 10},
            "creativity": {"score": creativity_points, "max": 15},
            "attempt_penalty": {"score": attempt_penalty, "max": 30},  # treat as deduction on UI
        }

        return jsonify({
            "success": success,
            "score": final_score,
            "guess": guess,
            "category": ai_category,
            "category_match": category_match,
            "color_match": color_match,
            "shape_match": shape_match,
            "style_score": style_score_raw,
            "background_score": background_score_raw,
            "creativity_score": creativity_score_raw,
            "expected_category": expected_category,
            "expected_colour": expected_colour,
            "target_id": target_prompt,
            "hint": hint,
            "hint_location": hint_location,
            "breakdown": breakdown
        })

    except Exception as e:
        print("Error in /submit:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


# -------------------- MAIN -------------------- #
if __name__ == "__main__":
    # Clear today's target so we always generate a fresh one on startup
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)

