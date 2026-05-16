"""
Google Maps enrichment: given a free-form address, return the same geographic
features needed for the model (suburb, distance to CBD, distance to nearest
train station, primary/secondary school counts within 2km).
"""
import math
import os
import time
import requests

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# Melbourne CBD Geo anchor point (GPO)
MELB_CBD = (-37.8136, 144.9631)

# Set school search radius to 2km, which is a typical primary school catchment size
SCHOOL_RADIUS_M = 2000

# Only accepts suburbs that were in the training data.
VALID_SUBURBS = {"Malvern", "Pakenham", "Ringwood"}


class EnrichmentError(Exception):
    """Raised when an address can't be geocoded."""


# Get the API key from an environment variable, if exists or raise an error.
def _api_key():
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise EnrichmentError(
            "GOOGLE_API_KEY is not configured on the server."
        )
    return key


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points in kilometres."""
    R = 6371
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2 +
         math.cos(lat1 * p) * math.cos(lat2 * p) *
         math.sin((lon2 - lon1) * p / 2) ** 2)
    return round(2 * R * math.asin(math.sqrt(a)), 2)


def geocode(address):
    """
    Resolve a free-form address to (lat, lon, detected_suburb).

    The suburb comes from Google's `locality` address component, which is the
    correct field for Australian suburbs in the geocoder response.
    """
    r = requests.get(
        GEOCODE_URL,
        params={"address": address + ", Australia", "key": _api_key()},
        timeout=10,
    )
    data = r.json()
    if data.get("status") != "OK" or not data.get("results"):
        status = data.get("status", "UNKNOWN")
        msg = data.get("error_message", "no results")
        raise EnrichmentError(f"Could not find that address ({status}: {msg}).")

    result = data["results"][0]
    loc = result["geometry"]["location"]

    detected_suburb = None
    for comp in result.get("address_components", []):
        if "locality" in comp.get("types", []):
            detected_suburb = comp["long_name"]
            break

    return loc["lat"], loc["lng"], detected_suburb


def nearest_station_km(lat, lon):
    """Distance in km from (lat, lon) to the closest train station."""
    r = requests.get(
        NEARBY_URL,
        params={
            "location": f"{lat},{lon}",
            "rankby": "distance",
            "type": "train_station",
            "key": _api_key(),
        },
        timeout=10,
    )
    data = r.json()
    if data.get("status") != "OK" or not data.get("results"):
        raise EnrichmentError(
            f"Train station lookup failed: {data.get('status', 'UNKNOWN')}"
        )
    top = data["results"][0]
    s_lat = top["geometry"]["location"]["lat"]
    s_lon = top["geometry"]["location"]["lng"]
    return haversine_km(lat, lon, s_lat, s_lon)


def count_places(lat, lon, place_type, radius_m):
    """
    Count places of a given type within radius_m, walking up to 3 pages of
    results (Google's max).
    """
    count = 0
    page_token = None
    key = _api_key()
    for _ in range(3):
        params = {
            "location": f"{lat},{lon}",
            "radius": radius_m,
            "type": place_type,
            "key": key,
        }
        if page_token:
            # Google requires a short delay before a next_page_token is valid.
            time.sleep(2)
            params = {"pagetoken": page_token, "key": key}

        r = requests.get(NEARBY_URL, params=params, timeout=10)
        data = r.json()
        status = data.get("status")
        if status not in ("OK", "ZERO_RESULTS"):
            raise EnrichmentError(
                f"Place search ({place_type}) failed: {status}"
            )
        count += len(data.get("results", []))
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return count


def enrich_address(address):
    """
    Returns a dict with the geographic features the model was trained on,
    plus the suburb Google detected from the address for cross-check.
    """
    lat, lon, detected_suburb = geocode(address)
    return {
        "detectedSuburb": detected_suburb,
        "latitude": lat,
        "longitude": lon,
        "distanceToCbd": haversine_km(lat, lon, *MELB_CBD),
        "distanceToStation": nearest_station_km(lat, lon),
        "primarySchools": count_places(lat, lon, "primary_school", SCHOOL_RADIUS_M),
        "secondarySchools": count_places(lat, lon, "secondary_school", SCHOOL_RADIUS_M),
    }
