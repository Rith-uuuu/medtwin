# database.py — PostgreSQL connection for MedTwin

from sqlalchemy import (create_engine, Column, String, Float,
                        Integer, JSON, DateTime, Text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# ── Connection ────────────────────────────────────────────────────
DATABASE_URL = "postgresql://postgres:medtwin123@localhost:5432/medtwin"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ── Tables ────────────────────────────────────────────────────────
class PatientDB(Base):
    __tablename__ = "patients"

    id             = Column(String,  primary_key=True)
    name           = Column(String,  nullable=False)
    age            = Column(Integer, nullable=False)
    bmi            = Column(Float,   nullable=False)
    glucose        = Column(Float,   nullable=False)
    insulin        = Column(Float,   nullable=False)
    p3             = Column(Float,   nullable=False)
    Gb             = Column(Float,   nullable=False)
    Ib             = Column(Float,   nullable=False)
    baseline_mean  = Column(Float,   nullable=False)
    baseline_std   = Column(Float,   nullable=False)
    metrics        = Column(JSON,    nullable=False)
    cgm_G          = Column(JSON,    nullable=False)
    created_at     = Column(DateTime, default=datetime.utcnow)
    created_by     = Column(String,  nullable=True)

class CGMReadingDB(Base):
    __tablename__ = "cgm_readings"

    id          = Column(Integer, primary_key=True,
                         autoincrement=True)
    patient_id  = Column(String,  nullable=False)
    glucose     = Column(Float,   nullable=False)
    timestamp   = Column(String,  nullable=False)

# ── Create tables ─────────────────────────────────────────────────
def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created!")

# ── Helper functions ──────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_patient(patient_dict: dict, db):
    p = PatientDB(
        id            = patient_dict["id"],
        name          = patient_dict["name"],
        age           = patient_dict["age"],
        bmi           = patient_dict["bmi"],
        glucose       = patient_dict["glucose"],
        insulin       = patient_dict["insulin"],
        p3            = patient_dict["p3"],
        Gb            = patient_dict["Gb"],
        Ib            = patient_dict["Ib"],
        baseline_mean = patient_dict["baseline_mean"],
        baseline_std  = patient_dict["baseline_std"],
        metrics       = patient_dict["metrics"],
        cgm_G         = patient_dict["cgm_G"],
        created_at    = datetime.utcnow(),
        created_by    = patient_dict.get("created_by","")
    )
    db.add(p)
    db.commit()

def load_all_patients(db):
    rows = db.query(PatientDB).all()
    result = {}
    for r in rows:
        result[r.id] = {
            "id":            r.id,
            "name":          r.name,
            "age":           r.age,
            "bmi":           r.bmi,
            "glucose":       r.glucose,
            "insulin":       r.insulin,
            "p3":            r.p3,
            "Gb":            r.Gb,
            "Ib":            r.Ib,
            "baseline_mean": r.baseline_mean,
            "baseline_std":  r.baseline_std,
            "metrics":       r.metrics,
            "cgm_G":         r.cgm_G,
            "created_at":    r.created_at.isoformat()
                             if r.created_at else "",
            "created_by":    r.created_by or ""
        }
    return result

def delete_patient_db(patient_id: str, db):
    db.query(PatientDB).filter(
        PatientDB.id==patient_id).delete()
    db.query(CGMReadingDB).filter(
        CGMReadingDB.patient_id==patient_id).delete()
    db.commit()

def save_cgm_reading(patient_id: str,
                     glucose: float,
                     timestamp: str, db):
    r = CGMReadingDB(
        patient_id=patient_id,
        glucose=glucose,
        timestamp=timestamp)
    db.add(r)
    db.commit()

def load_cgm_readings(patient_id: str, db):
    rows = db.query(CGMReadingDB).filter(
        CGMReadingDB.patient_id==patient_id).all()
    return [{"glucose":r.glucose,
             "timestamp":r.timestamp,
             "index":i}
            for i,r in enumerate(rows)]

if __name__ == "__main__":
    init_db()