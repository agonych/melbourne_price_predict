"""
Flask entry point.
"""
import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from enrich import enrich_address, EnrichmentError, VALID_SUBURBS
from model import predict_price, VALID_PROPERTY_TYPES, VALID_SOLD_YEARS

# Loads .env in local dev. On Render the env vars come from the dashboard
# and load_dotenv is a no-op (no .env file present).
load_dotenv()

app = Flask(__name__)

# Home route
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        valid_property_types=sorted(VALID_PROPERTY_TYPES),
        valid_suburbs=sorted(VALID_SUBURBS),
        valid_sold_years=sorted(VALID_SOLD_YEARS, reverse=True),
    )

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # ---- Validate user input ----
    try:
        user_input = {
            "bedrooms": int(request.form["bedrooms"]),
            "bathrooms": int(request.form["bathrooms"]),
            "parkingSpaces": int(request.form["parkingSpaces"]),
            "landSize": (
                float(request.form["landSize"])
                if request.form.get("landSize") else None
            ),
            "propertyType": request.form["propertyType"].lower().strip(),
            "suburb": request.form["suburb"].strip(),
            "soldYear": int(request.form["soldYear"]),
        }
    except (KeyError, ValueError) as e:
        return jsonify({"ok": False, "error": f"Invalid input: {e}"}), 400

    if user_input["propertyType"] not in VALID_PROPERTY_TYPES:
        return jsonify({
            "ok": False,
            "error": (
                f"Invalid propertyType. Must be one of: "
                f"{', '.join(sorted(VALID_PROPERTY_TYPES))}."
            ),
        }), 400

    if user_input["suburb"] not in VALID_SUBURBS:
        return jsonify({
            "ok": False,
            "error": (
                f"Invalid suburb. Must be one of: "
                f"{', '.join(sorted(VALID_SUBURBS))}."
            ),
        }), 400

    if user_input["soldYear"] not in VALID_SOLD_YEARS:
        return jsonify({
            "ok": False,
            "error": (
                f"Invalid sold year. Must be one of: "
                f"{', '.join(str(y) for y in sorted(VALID_SOLD_YEARS))}."
            ),
        }), 400

    address = request.form.get("address", "").strip()
    if not address:
        return jsonify({"ok": False, "error": "Address is required."}), 400

    # ---- Enrich + predict ----
    try:
        enrichment = enrich_address(address)

        # The geocoded suburb should match the dropdown choice.
        detected = enrichment.get("detectedSuburb")
        if detected != user_input["suburb"]:
            return jsonify({
                "ok": False,
                "error": (
                    f"Address geocoded to '{detected or 'unknown'}' but you "
                    f"selected '{user_input['suburb']}'. Please check the "
                    f"address and the suburb dropdown match."
                ),
            }), 400

        price = predict_price(user_input, enrichment)
    except EnrichmentError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "ok": False,
            "error": "Internal server error. Try again in a moment.",
        }), 500

    return jsonify({
        "ok": True,
        "predicted_price": round(price),
        "predicted_price_display": f"${price:,.0f}",
        "enrichment": enrichment,
    })


if __name__ == "__main__":
    # Local dev only — Render uses gunicorn (see render.yaml).
    app.run(debug=True, host="127.0.0.1", port=5000)
