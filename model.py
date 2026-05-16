"""
Model loading and prediction.

The pickled object is the full sklearn Pipeline (preprocessor + GradientBoosting).
It needs a DataFrame with the correct column names and will handle scaling
internally. No need to fit any transformer here.
"""
from pathlib import Path
import joblib
import pandas as pd

# Must exactly match the column order the model was trained on.
FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "parkingSpaces",
    "landSize",
    "distanceToCbd",
    "distanceToStation",
    "primarySchools",
    "secondarySchools",
    "soldYear",
    "propertyType_house",
    "propertyType_townhouse",
    "propertyType_unit",
    "suburb_Pakenham",
    "suburb_Ringwood",
    "landSizeMissing",
]

# Acceptable propterty types
VALID_PROPERTY_TYPES = {"apartment", "house", "townhouse", "unit"}
# Acceptable years
VALID_SOLD_YEARS = {2025, 2026}

_model = None


def get_model():
    """Lazy-load the model on first prediction (saves cold-start time)."""
    global _model
    if _model is None:
        _model = joblib.load(Path(__file__).parent / "model.pkl")
    return _model


def build_feature_row(user_input, enrichment):
    """
    Build a single-row DataFrame in the exact column order the model expects.

    user_input keys:  bedrooms, bathrooms, parkingSpaces, landSize,
                      propertyType, suburb, soldYear
    enrichment keys:  distanceToCbd, distanceToStation,
                      primarySchools, secondarySchools
    """
    land_size = user_input.get("landSize")
    land_size_value = land_size if land_size is not None else 0
    land_size_missing = 1 if land_size is None else 0

    # Start with every feature at zero, then fill in what we know.
    row = {col: 0 for col in FEATURE_COLUMNS}
    row.update({
        "bedrooms": user_input["bedrooms"],
        "bathrooms": user_input["bathrooms"],
        "parkingSpaces": user_input["parkingSpaces"],
        "landSize": land_size_value,
        "distanceToCbd": enrichment["distanceToCbd"],
        "distanceToStation": enrichment["distanceToStation"],
        "primarySchools": enrichment["primarySchools"],
        "secondarySchools": enrichment["secondarySchools"],
        # soldYear in the model is binary: 1 for 2026, 0 for 2025
        "soldYear": 1 if user_input["soldYear"] == 2026 else 0,
        "landSizeMissing": land_size_missing,
    })

    # One-hot for propertyType (apartment is the dropped reference)
    pt = user_input["propertyType"]
    if pt == "house":
        row["propertyType_house"] = 1
    elif pt == "townhouse":
        row["propertyType_townhouse"] = 1
    elif pt == "unit":
        row["propertyType_unit"] = 1

    # One-hot for suburb (Malvern is the dropped reference)
    suburb = user_input["suburb"]
    if suburb == "Pakenham":
        row["suburb_Pakenham"] = 1
    elif suburb == "Ringwood":
        row["suburb_Ringwood"] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_price(user_input, enrichment):
    """Return a non-negative predicted price as a float."""
    features = build_feature_row(user_input, enrichment)
    raw = float(get_model().predict(features)[0])
    # Trees can occasionally predict slightly negative for edge cases —
    # clamp at zero rather than show "$-12,000" in the UI.
    return max(0.0, raw)
