# MedTwin FastAPI Backend — Complete API v3.0 with PostgreSQL + Dexcom
# main.py — C:\Users\ravichandran\MT\main.py

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import numpy as np
from scipy.integrate import solve_ivp
import warnings
import uuid

from database import (init_db, save_patient, load_all_patients,
                      delete_patient_db, save_cgm_reading,
                      load_cgm_readings, SessionLocal)
from dexcom import (get_auth_url, exchange_code,
                    fetch_egvs, test_sandbox_connection)

warnings.filterwarnings('ignore')

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="MedTwin API",
    description="Digital Twin for Indian T2D Patients",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────
SECRET_KEY  = "medtwin-secret-key-change-in-production"
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# ── Memory cache ──────────────────────────────────────────────────
patients  = {}
cgm_store = {}

# ── Doctors ───────────────────────────────────────────────────────
DOCTORS = {
    "doctor1": {
        "username":        "doctor1",
        "full_name":       "Dr. Priya Raman",
        "email":           "priya@medtwin.in",
        "hashed_password": pwd_context.hash("password123"),
        "role":            "doctor"
    },
    "admin": {
        "username":        "admin",
        "full_name":       "Admin User",
        "email":           "admin@medtwin.in",
        "hashed_password": pwd_context.hash("admin123"),
        "role":            "admin"
    }
}

# ════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    init_db()
    global patients, cgm_store
    db = SessionLocal()
    try:
        patients = load_all_patients(db)
        for pid in patients:
            cgm_store[pid] = load_cgm_readings(pid, db)
        print(f"Loaded {len(patients)} patients from PostgreSQL")
    finally:
        db.close()

# ════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ════════════════════════════════════════════════════════

class PatientInput(BaseModel):
    name:    str
    age:     int
    bmi:     float
    glucose: float
    insulin: float
    outcome: Optional[int] = 1

class TreatmentInput(BaseModel):
    patient_id:       str
    metformin_effect: float = 0.0
    diet_effect:      float = 0.0
    exercise_effect:  float = 0.0
    days:             int   = 30

class CGMReading(BaseModel):
    glucose:   float
    timestamp: Optional[str] = None

class CGMBatch(BaseModel):
    readings: List[CGMReading]

class Token(BaseModel):
    access_token: str
    token_type:   str
    doctor_name:  str
    role:         str

# ════════════════════════════════════════════════════════
# AUTH HELPERS
# ════════════════════════════════════════════════════════

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire    = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_doctor(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(
            token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in DOCTORS:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return DOCTORS[username]

# ════════════════════════════════════════════════════════
# SCIENCE HELPERS
# ════════════════════════════════════════════════════════

MEALS = [(480,60),(780,85),(1260,60)]

def estimate_params(glucose, insulin, bmi, age):
    p3 = 0.000013 * (
        np.exp(-insulin/150) *
        np.exp(-glucose/200) *
        np.exp(-bmi/60) *
        np.exp(-age/100))
    return (float(np.clip(p3,1e-7,0.000013)),
            float(np.clip(glucose,70,200)),
            float(np.clip(insulin,5,50)))

def simulate_glucose(p3, Gb, Ib, days=1,
                     metformin=0, diet=0, exercise=0):
    total     = days * 1440
    all_meals = [(d*1440+tm,c)
                 for d in range(days) for tm,c in MEALS]
    def bergman(t, y):
        G,X,I,Ra = y
        mi     = sum((c*(1-diet)/30)
                     for tm,c in all_meals if tm<=t<=tm+30)
        Gb_eff = Gb*(1-metformin)
        p3_eff = p3*(1+exercise)
        dRa = -0.05*Ra + mi
        dG  = -0.028*(G-Gb_eff) - X*G + 0.01*Ra
        dX  = -0.025*X + p3_eff*(I-Ib)
        dI  = -0.093*(I-Ib) + 0.005*Ra
        return [dG,dX,dI,dRa]
    sol = solve_ivp(bergman,[0,total],[Gb,0,Ib,0],
                    max_step=5,dense_output=True)
    t = np.arange(0,total,5)
    G = np.clip(sol.sol(t)[0],40,400)
    return t.tolist(), G.tolist()

def compute_metrics(G):
    G     = np.clip(np.array(G,dtype=float),40,400)
    tir   = float(((G>=70)&(G<=180)).mean()*100)
    tab   = float((G>180).mean()*100)
    tbel  = float((G<70).mean()*100)
    gmean = float(G.mean())
    return {
        "tir":          round(tir,1),
        "time_above":   round(tab,1),
        "time_below":   round(tbel,1),
        "mean_glucose": round(gmean,1),
        "hba1c":        round((gmean+46.7)/28.7,2)
    }

def detect_anomalies(G, baseline_mean, baseline_std):
    G         = np.array(G)
    threshold = baseline_mean + 2.5*baseline_std
    alerts    = []
    window    = 84
    for i in range(window, len(G)):
        if G[i] > threshold:
            if not any(abs(a['index']-i)<12 for a in alerts
                       if a['type']=='Acute'):
                alerts.append({
                    "type":    "Acute",
                    "index":   i,
                    "time_hr": round(i*5/60,1),
                    "glucose": round(float(G[i]),1),
                    "message": (f"Glucose {G[i]:.0f} exceeds "
                                f"personal threshold "
                                f"{threshold:.0f} mg/dL")
                })
        recent = G[i-window:i].mean()
        if recent > baseline_mean + 1.5*baseline_std:
            if not any(abs(a['index']-i)<window for a in alerts
                       if a['type']=='Trend'):
                alerts.append({
                    "type":    "Trend",
                    "index":   i,
                    "time_hr": round(i*5/60,1),
                    "glucose": round(float(recent),1),
                    "message": (f"7-hour mean {recent:.0f} — "
                                f"sustained elevation above "
                                f"personal baseline")
                })
    return alerts[:10]

# ════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ════════════════════════════════════════════════════════

@app.post("/auth/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    doctor = DOCTORS.get(form.username)
    if not doctor or not verify_password(
            form.password, doctor["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password")
    token = create_access_token({"sub": doctor["username"]})
    return {
        "access_token": token,
        "token_type":   "bearer",
        "doctor_name":  doctor["full_name"],
        "role":         doctor["role"]
    }

@app.get("/auth/me")
def get_me(doctor=Depends(get_current_doctor)):
    return {
        "username":  doctor["username"],
        "full_name": doctor["full_name"],
        "email":     doctor["email"],
        "role":      doctor["role"]
    }

# ════════════════════════════════════════════════════════
# ROOT + STATS
# ════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "name":     "MedTwin API",
        "version":  "3.0.0",
        "status":   "running",
        "database": "PostgreSQL",
        "patients": len(patients),
    }

@app.get("/stats")
def get_stats(doctor=Depends(get_current_doctor)):
    if not patients:
        return {
            "total_patients":    0,
            "avg_tir":           0,
            "total_alerts":      0,
            "critical_patients": 0,
            "needs_attention":   []
        }
    all_tir    = [p["metrics"]["tir"] for p in patients.values()]
    all_alerts = sum(
        len(detect_anomalies(
            p["cgm_G"],p["baseline_mean"],p["baseline_std"]))
        for p in patients.values())
    critical = sum(
        1 for p in patients.values()
        if p["metrics"]["tir"] < 50)
    return {
        "total_patients":    len(patients),
        "avg_tir":           round(float(np.mean(all_tir)),1),
        "total_alerts":      all_alerts,
        "critical_patients": critical,
        "needs_attention": [
            {"id":p["id"],"name":p["name"],
             "tir":p["metrics"]["tir"]}
            for p in patients.values()
            if p["metrics"]["tir"] < 50
        ]
    }

# ════════════════════════════════════════════════════════
# PATIENT ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/patients")
def list_patients(doctor=Depends(get_current_doctor)):
    return {
        "count":   len(patients),
        "doctor":  doctor["full_name"],
        "patients": [
            {
                "id":    p["id"],
                "name":  p["name"],
                "age":   p["age"],
                "tir":   p["metrics"]["tir"],
                "hba1c": p["metrics"]["hba1c"],
                "alerts": len(detect_anomalies(
                    p["cgm_G"],
                    p["baseline_mean"],
                    p["baseline_std"]))
            }
            for p in patients.values()
        ]
    }

@app.post("/patient")
def create_patient(data: PatientInput,
                   doctor=Depends(get_current_doctor)):
    patient_id  = str(uuid.uuid4())[:8]
    p3, Gb, Ib  = estimate_params(
        data.glucose, data.insulin, data.bmi, data.age)
    t, G        = simulate_glucose(p3, Gb, Ib, days=1)
    metrics     = compute_metrics(G)
    bm          = float(np.mean(G))
    bs          = float(np.std(G))

    patient = {
        "id":            patient_id,
        "name":          data.name,
        "age":           data.age,
        "bmi":           data.bmi,
        "glucose":       data.glucose,
        "insulin":       data.insulin,
        "p3":            p3,
        "Gb":            Gb,
        "Ib":            Ib,
        "baseline_mean": bm,
        "baseline_std":  bs,
        "metrics":       metrics,
        "cgm_G":         G,
        "created_at":    datetime.utcnow().isoformat(),
        "created_by":    doctor["username"],
    }

    patients[patient_id]  = patient
    cgm_store[patient_id] = []

    db = SessionLocal()
    try:
        save_patient(patient, db)
    finally:
        db.close()

    return {
        "patient_id":  patient_id,
        "message":     f"Twin created for {data.name}",
        "created_by":  doctor["full_name"],
        "parameters":  {
            "p3": round(p3*1e6,4),
            "Gb": round(Gb,1),
            "Ib": round(Ib,1)
        },
        "metrics": metrics
    }

@app.get("/patient/{patient_id}")
def get_patient(patient_id: str,
                doctor=Depends(get_current_doctor)):
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    p = patients[patient_id]
    return {
        "id":       p["id"],
        "name":     p["name"],
        "age":      p["age"],
        "bmi":      p["bmi"],
        "glucose":  p["glucose"],
        "insulin":  p["insulin"],
        "parameters": {
            "p3": round(p["p3"]*1e6,4),
            "Gb": round(p["Gb"],1),
            "Ib": round(p["Ib"],1)
        },
        "metrics":      p["metrics"],
        "created_at":   p.get("created_at",""),
        "cgm_readings": len(cgm_store.get(patient_id,[]))
    }

@app.delete("/patient/{patient_id}")
def delete_patient(patient_id: str,
                   doctor=Depends(get_current_doctor)):
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    name = patients[patient_id]["name"]
    del patients[patient_id]
    cgm_store.pop(patient_id, None)
    db = SessionLocal()
    try:
        delete_patient_db(patient_id, db)
    finally:
        db.close()
    return {"message": f"Patient {name} deleted"}

# ════════════════════════════════════════════════════════
# SIMULATION ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/patient/{patient_id}/simulate")
def simulate_patient(patient_id: str,
                     doctor=Depends(get_current_doctor)):
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    p    = patients[patient_id]
    t, G = simulate_glucose(p["p3"], p["Gb"], p["Ib"], days=1)
    return {
        "patient_id": patient_id,
        "time":       t[::6],
        "glucose":    [round(g,1) for g in G[::6]],
        "metrics":    compute_metrics(G)
    }

@app.post("/patient/{patient_id}/treatment")
def simulate_treatment(patient_id: str,
                       data: TreatmentInput,
                       doctor=Depends(get_current_doctor)):
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    p = patients[patient_id]
    np.random.seed(42)
    all_tir, all_hba1c, all_mean = [], [], []
    for _ in range(50):
        p3_s = p["p3"] * np.random.uniform(0.9,1.1)
        Gb_s = p["Gb"] + np.random.uniform(-5,5)
        _,G  = simulate_glucose(
            p3_s, Gb_s, p["Ib"],
            days=data.days,
            metformin=data.metformin_effect,
            diet=data.diet_effect,
            exercise=data.exercise_effect)
        m = compute_metrics(G)
        all_tir.append(m["tir"])
        all_hba1c.append(m["hba1c"])
        all_mean.append(m["mean_glucose"])
    return {
        "patient_id": patient_id,
        "treatment": {
            "metformin_effect": data.metformin_effect,
            "diet_effect":      data.diet_effect,
            "exercise_effect":  data.exercise_effect,
            "days":             data.days
        },
        "results": {
            "tir_mean":     round(float(np.mean(all_tir)),1),
            "tir_std":      round(float(np.std(all_tir)),1),
            "hba1c_mean":   round(float(np.mean(all_hba1c)),2),
            "hba1c_std":    round(float(np.std(all_hba1c)),2),
            "mean_glucose": round(float(np.mean(all_mean)),1),
            "prob_target":  round(
                float((np.array(all_tir)>=70).mean()*100),1)
        }
    }

@app.get("/patient/{patient_id}/alerts")
def get_alerts(patient_id: str,
               doctor=Depends(get_current_doctor)):
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    p      = patients[patient_id]
    alerts = detect_anomalies(
        p["cgm_G"], p["baseline_mean"], p["baseline_std"])
    return {
        "patient_id":  patient_id,
        "alert_count": len(alerts),
        "alerts":      alerts
    }

# ════════════════════════════════════════════════════════
# CGM ENDPOINTS
# ════════════════════════════════════════════════════════

@app.post("/patient/{patient_id}/cgm")
def add_cgm_reading(patient_id: str,
                    reading: CGMReading,
                    doctor=Depends(get_current_doctor)):
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    if not 40 <= reading.glucose <= 400:
        raise HTTPException(400,
            f"Invalid glucose: {reading.glucose}")
    timestamp = reading.timestamp or datetime.utcnow().isoformat()
    entry = {
        "glucose":   round(reading.glucose,1),
        "timestamp": timestamp,
        "index":     len(cgm_store[patient_id])
    }
    cgm_store[patient_id].append(entry)
    patients[patient_id]["cgm_G"].append(reading.glucose)
    recent_G = [r["glucose"]
                for r in cgm_store[patient_id][-288:]]
    if len(recent_G) >= 12:
        patients[patient_id]["metrics"] = compute_metrics(recent_G)
    db = SessionLocal()
    try:
        save_cgm_reading(patient_id, reading.glucose,
                         timestamp, db)
    finally:
        db.close()
    p         = patients[patient_id]
    threshold = p["baseline_mean"] + 2.5*p["baseline_std"]
    alert     = None
    if reading.glucose > threshold:
        alert = {
            "type":      "Acute",
            "glucose":   reading.glucose,
            "threshold": round(threshold,1),
            "message":   (f"Glucose {reading.glucose:.0f} "
                          f"exceeds personal threshold "
                          f"{threshold:.0f} mg/dL")
        }
    return {
        "patient_id":      patient_id,
        "reading_saved":   entry,
        "total_readings":  len(cgm_store[patient_id]),
        "alert":           alert,
        "current_metrics": patients[patient_id]["metrics"]
    }

@app.post("/patient/{patient_id}/cgm/batch")
def add_cgm_batch(patient_id: str,
                  data: CGMBatch,
                  doctor=Depends(get_current_doctor)):
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    added     = 0
    alerts    = []
    p         = patients[patient_id]
    threshold = p["baseline_mean"] + 2.5*p["baseline_std"]
    db        = SessionLocal()
    try:
        for reading in data.readings:
            if not 40 <= reading.glucose <= 400:
                continue
            timestamp = (reading.timestamp or
                         datetime.utcnow().isoformat())
            entry = {
                "glucose":   round(reading.glucose,1),
                "timestamp": timestamp,
                "index":     len(cgm_store[patient_id])
            }
            cgm_store[patient_id].append(entry)
            patients[patient_id]["cgm_G"].append(reading.glucose)
            save_cgm_reading(patient_id, reading.glucose,
                             timestamp, db)
            if reading.glucose > threshold:
                alerts.append({
                    "glucose":   reading.glucose,
                    "timestamp": timestamp
                })
            added += 1
    finally:
        db.close()
    recent_G = [r["glucose"]
                for r in cgm_store[patient_id][-288:]]
    if len(recent_G) >= 12:
        patients[patient_id]["metrics"] = compute_metrics(recent_G)
    return {
        "patient_id":      patient_id,
        "readings_added":  added,
        "total_readings":  len(cgm_store[patient_id]),
        "alerts_fired":    len(alerts),
        "updated_metrics": patients[patient_id]["metrics"]
    }

@app.get("/patient/{patient_id}/history")
def get_history(patient_id: str, hours: int=24,
                doctor=Depends(get_current_doctor)):
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    readings = cgm_store.get(patient_id,[])
    if len(readings)==0:
        p    = patients[patient_id]
        t, G = simulate_glucose(
            p["p3"], p["Gb"], p["Ib"],
            days=max(1,hours//24))
        points = hours*12
        return {
            "patient_id":  patient_id,
            "source":      "simulated",
            "hours":       hours,
            "data_points": min(points,len(t)),
            "time":        t[:points:6],
            "glucose":     [round(g,1) for g in G[:points:6]],
            "metrics":     compute_metrics(G[:points])
        }
    limit   = hours*12
    recent  = readings[-limit:]
    glucose = [r["glucose"]   for r in recent]
    times   = [r["timestamp"] for r in recent]
    return {
        "patient_id":  patient_id,
        "source":      "real_cgm",
        "hours":       hours,
        "data_points": len(recent),
        "timestamps":  times,
        "glucose":     glucose,
        "metrics":     compute_metrics(glucose) if glucose else {}
    }

@app.get("/patient/{patient_id}/summary")
def get_summary(patient_id: str,
                doctor=Depends(get_current_doctor)):
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    p        = patients[patient_id]
    readings = cgm_store.get(patient_id,[])
    alerts   = detect_anomalies(
        p["cgm_G"], p["baseline_mean"], p["baseline_std"])
    return {
        "patient_id":   patient_id,
        "name":         p["name"],
        "age":          p["age"],
        "bmi":          p["bmi"],
        "created_at":   p.get("created_at",""),
        "twin_parameters": {
            "p3_x1e6":       round(p["p3"]*1e6,4),
            "Gb_mg_dL":      round(p["Gb"],1),
            "Ib_uU_mL":      round(p["Ib"],1),
            "baseline_mean": round(p["baseline_mean"],1),
            "baseline_std":  round(p["baseline_std"],1),
            "alert_threshold": round(
                p["baseline_mean"]+2.5*p["baseline_std"],1)
        },
        "current_metrics":    p["metrics"],
        "total_cgm_readings": len(readings),
        "active_alerts":      len(alerts),
        "alerts":             alerts[:3],
        "clinical_flags": {
            "needs_intervention": p["metrics"]["tir"] < 50,
            "hyperglycemia_risk": p["metrics"]["time_above"] > 25,
            "hypoglycemia_risk":  p["metrics"]["time_below"] > 4,
            "hba1c_above_target": p["metrics"]["hba1c"] > 7.0
        }
    }

# ════════════════════════════════════════════════════════
# DEXCOM ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/dexcom/test")
def test_dexcom(doctor=Depends(get_current_doctor)):
    result = test_sandbox_connection()
    return result

@app.get("/patient/{patient_id}/dexcom/connect")
def connect_dexcom(patient_id: str,
                   doctor=Depends(get_current_doctor)):
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    auth_url = get_auth_url(patient_id)
    return {
        "patient_id": patient_id,
        "auth_url":   auth_url,
        "message":    "Open this URL to connect Dexcom account"
    }

@app.get("/auth/dexcom/callback")
def dexcom_callback(code: str, state: str):
    try:
        patient_id = state
        exchange_code(code, patient_id)
        readings   = fetch_egvs(patient_id, hours=24)
        db         = SessionLocal()
        try:
            if patient_id not in cgm_store:
                cgm_store[patient_id] = []
            for r in readings:
                cgm_store[patient_id].append({
                    "glucose":   r["glucose"],
                    "timestamp": r["timestamp"],
                    "index":     len(cgm_store[patient_id])
                })
                save_cgm_reading(
                    patient_id, r["glucose"],
                    r["timestamp"], db)
        finally:
            db.close()
        return {
            "status":          "connected",
            "patient_id":      patient_id,
            "readings_synced": len(readings),
            "message":         f"Dexcom connected! {len(readings)} readings synced."
        }
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/patient/{patient_id}/dexcom/sync")
def sync_dexcom(patient_id: str,
                doctor=Depends(get_current_doctor)):
    if patient_id not in patients:
        raise HTTPException(404, "Patient not found")
    try:
        readings = fetch_egvs(patient_id, hours=6)
        db       = SessionLocal()
        added    = 0
        try:
            if patient_id not in cgm_store:
                cgm_store[patient_id] = []
            for r in readings:
                cgm_store[patient_id].append({
                    "glucose":   r["glucose"],
                    "timestamp": r["timestamp"],
                    "index":     len(cgm_store[patient_id])
                })
                patients[patient_id]["cgm_G"].append(r["glucose"])
                save_cgm_reading(
                    patient_id, r["glucose"],
                    r["timestamp"], db)
                added += 1
        finally:
            db.close()
        recent_G = [x["glucose"]
                    for x in cgm_store[patient_id][-288:]]
        if len(recent_G) >= 12:
            patients[patient_id]["metrics"] = \
                compute_metrics(recent_G)
        return {
            "patient_id":      patient_id,
            "readings_synced": added,
            "total_readings":  len(cgm_store[patient_id]),
            "updated_metrics": patients[patient_id]["metrics"]
        }
    except Exception as e:
        raise HTTPException(400, f"Dexcom sync failed: {str(e)}")