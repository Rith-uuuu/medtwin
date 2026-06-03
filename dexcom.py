# dexcom.py — Dexcom Sandbox Integration for MedTwin

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID     = os.getenv("DEXCOM_CLIENT_ID")
CLIENT_SECRET = os.getenv("DEXCOM_CLIENT_SECRET")
REDIRECT_URI  = os.getenv("DEXCOM_REDIRECT_URI")
BASE_URL      = os.getenv("DEXCOM_BASE_URL",
                          "https://sandbox-api.dexcom.com")

# ── Token store (in production store in DB per patient) ───────────
dexcom_tokens = {}  # patient_id → {access_token, refresh_token}

# ── Step 1: Generate auth URL ─────────────────────────────────────
def get_auth_url(patient_id: str) -> str:
    """Doctor clicks this link to connect patient's Dexcom"""
    return (
        f"{BASE_URL}/v2/oauth2/login"
        f"?client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=offline_access"
        f"&state={patient_id}"
    )

# ── Step 2: Exchange code for token ───────────────────────────────
def exchange_code(code: str, patient_id: str) -> dict:
    """Called after patient authorizes — exchange code for token"""
    r = requests.post(
        f"{BASE_URL}/v2/oauth2/token",
        data={
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code":          code,
            "grant_type":    "authorization_code",
            "redirect_uri":  REDIRECT_URI,
        }
    )
    if r.status_code != 200:
        raise Exception(f"Token exchange failed: {r.text}")

    token_data = r.json()
    dexcom_tokens[patient_id] = {
        "access_token":  token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "expires_at":    datetime.utcnow() + timedelta(
            seconds=token_data.get("expires_in", 7200))
    }
    return dexcom_tokens[patient_id]

# ── Step 3: Refresh token if expired ─────────────────────────────
def refresh_token(patient_id: str) -> str:
    """Refresh access token using refresh token"""
    if patient_id not in dexcom_tokens:
        raise Exception("No token found for patient")

    token_data = dexcom_tokens[patient_id]

    # Check if still valid
    if datetime.utcnow() < token_data["expires_at"]:
        return token_data["access_token"]

    # Refresh
    r = requests.post(
        f"{BASE_URL}/v2/oauth2/token",
        data={
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": token_data["refresh_token"],
            "grant_type":    "refresh_token",
            "redirect_uri":  REDIRECT_URI,
        }
    )
    if r.status_code != 200:
        raise Exception(f"Token refresh failed: {r.text}")

    new_token = r.json()
    dexcom_tokens[patient_id] = {
        "access_token":  new_token["access_token"],
        "refresh_token": new_token.get(
            "refresh_token",
            token_data["refresh_token"]),
        "expires_at":    datetime.utcnow() + timedelta(
            seconds=new_token.get("expires_in", 7200))
    }
    return dexcom_tokens[patient_id]["access_token"]

# ── Step 4: Fetch CGM readings ────────────────────────────────────
def get_data_range(patient_id: str) -> dict:
    """Get the available date range for this sandbox user"""
    access_token = refresh_token(patient_id)
    r = requests.get(
        f"{BASE_URL}/v3/users/self/dataRange",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    if r.status_code != 200:
        raise Exception(f"dataRange failed: {r.text}")
    return r.json()

def fetch_egvs(patient_id: str, hours: int = 24) -> list:
    """Fetch estimated glucose values from Dexcom"""
    access_token = refresh_token(patient_id)

    # First get the actual data range for this sandbox user
    try:
        data_range  = get_data_range(patient_id)
        egvs_range  = data_range.get("egvs", {})
        end_str     = egvs_range.get("end", {}).get(
            "systemTime", "")
        if end_str:
            end_date   = datetime.strptime(
                end_str[:19], "%Y-%m-%dT%H:%M:%S")
            start_date = end_date - timedelta(hours=hours)
        else:
            end_date   = datetime.utcnow()
            start_date = end_date - timedelta(hours=hours)
    except:
        end_date   = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours)

    r = requests.get(
        f"{BASE_URL}/v3/users/self/egvs",
        headers={"Authorization": f"Bearer {access_token}"},
        params={
            "startDate": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "endDate":   end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )

    if r.status_code != 200:
        raise Exception(f"EGV fetch failed: {r.text}")

    data     = r.json()
    readings = []
    for egv in data.get("egvs", []):
        val = egv.get("value")
        if val and val not in ["low", "high"]:
            try:
                readings.append({
                    "glucose":   float(val),
                    "timestamp": egv["systemTime"],
                    "trend":     egv.get("trend",""),
                })
            except:
                pass

    return readings

# ── Sandbox test function ─────────────────────────────────────────
def test_sandbox_connection() -> dict:
    """Test if Dexcom sandbox credentials are working"""
    auth_url = get_auth_url("test_patient")
    return {
        "status":   "credentials loaded",
        "auth_url": auth_url,
        "client_id": CLIENT_ID[:8] + "...",
        "sandbox":  BASE_URL
    }

if __name__ == "__main__":
    result = test_sandbox_connection()
    print("Dexcom sandbox connection test:")
    for k, v in result.items():
        print(f"  {k}: {v}")