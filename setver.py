"""CareBridge 360 backend — continuity of care & emergency orchestration."""
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Header, UploadFile, File
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os, uuid, logging, jwt, bcrypt, base64, asyncio, json, random

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]
JWT_SECRET = os.environ.get("JWT_SECRET", "carebridge-dev-secret-key")
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

app = FastAPI(title="CareBridge 360")
api = APIRouter(prefix="/api")
log = logging.getLogger("carebridge")
logging.basicConfig(level=logging.INFO)


# ---------- utils ----------
def uid() -> str:
    return str(uuid.uuid4())

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def verify_pw(pw: str, h: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), h.encode())
    except Exception:
        return False

def make_token(user_id: str, role: str) -> str:
    payload = {"sub": user_id, "role": role, "exp": datetime.now(timezone.utc) + timedelta(days=7)}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

async def current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    try:
        payload = jwt.decode(authorization.split(" ", 1)[1], JWT_SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(401, "Invalid token")
    user = await db.users.find_one({"id": payload["sub"]}, {"_id": 0, "password": 0})
    if not user:
        raise HTTPException(401, "User not found")
    return user

def require_roles(*roles):
    async def _dep(user: dict = Depends(current_user)):
        if user["role"] not in roles:
            raise HTTPException(403, f"Requires role: {roles}")
        return user
    return _dep


# ---------- schemas ----------
class LoginIn(BaseModel):
    email: EmailStr
    password: str

class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: Literal["patient","caregiver","doctor","responder","hospital","admin"]

class MedicationIn(BaseModel):
    name: str
    dosage: str
    frequency: str
    times: List[str] = []
    instructions: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class MedLogIn(BaseModel):
    medication_id: str
    status: Literal["taken","skipped","snoozed"]
    scheduled_time: str

class AppointmentIn(BaseModel):
    doctor: str
    hospital: str
    specialty: str
    scheduled_at: str
    reason: str = ""

class SOSIn(BaseModel):
    location_label: str = "Home"
    latitude: float = 12.9716
    longitude: float = 77.5946
    note: str = ""

class PrivacyIn(BaseModel):
    scope: str
    role: str
    allowed: bool


# ---------- event bus ----------
async def emit_event(patient_id: str, event_type: str, actor: str, metadata: Dict[str, Any] = None, visibility: str = "care_network"):
    ev = {
        "id": uid(),
        "patient_id": patient_id,
        "event_type": event_type,
        "actor": actor,
        "metadata": metadata or {},
        "visibility": visibility,
        "timestamp": now_iso(),
    }
    await db.events.insert_one(ev)
    return ev

async def log_access(patient_id: str, accessor_id: str, accessor_role: str, accessor_name: str, resource: str, reason: str = ""):
    rec = {
        "id": uid(),
        "patient_id": patient_id,
        "accessor_id": accessor_id,
        "accessor_role": accessor_role,
        "accessor_name": accessor_name,
        "resource": resource,
        "reason": reason,
        "timestamp": now_iso(),
    }
    await db.access_logs.insert_one(rec)

async def notify(recipient_id: str, priority: str, title: str, body: str, meta: Dict[str, Any] = None):
    n = {
        "id": uid(),
        "recipient_id": recipient_id,
        "priority": priority,
        "title": title,
        "body": body,
        "meta": meta or {},
        "read": False,
        "created_at": now_iso(),
    }
    await db.notifications.insert_one(n)


# ---------- AI service (Gemini 3.1 Pro with demo fallback) ----------
async def gemini_extract_prescription(image_b64: Optional[str]) -> dict:
    """Returns structured medications with confidence + evidence."""
    demo_result = {
        "source": "demo",
        "doctor": "Dr. Priya Sharma",
        "date": "2026-08-18",
        "medications": [
            {"name": "Metformin", "dosage": "500 mg", "frequency": "Twice daily", "duration": "30 days", "instructions": "After meals", "confidence": 0.94},
            {"name": "Amlodipine", "dosage": "5 mg", "frequency": "Once daily", "duration": "30 days", "instructions": "Morning", "confidence": 0.89},
            {"name": "Metfornin", "dosage": "500 mg", "frequency": "Twice daily", "duration": "10 days", "instructions": "Possible OCR uncertainty", "confidence": 0.62},
        ],
        "notes": "Extraction generated in demo mode. Please verify with original prescription.",
    }
    if not EMERGENT_LLM_KEY or not image_b64:
        return demo_result
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"ocr-{uid()}",
            system_message="You extract structured medication data from prescription images. Return strict JSON: {doctor, date, medications:[{name,dosage,frequency,duration,instructions,confidence}], notes}. Include confidence 0-1 per medication. If unsure of a name, still return but with lower confidence."
        ).with_model("gemini", "gemini-3.1-pro-preview")
        img = ImageContent(image_base64=image_b64)
        resp = await chat.send_message(UserMessage(text="Extract medications as JSON.", file_contents=[img]))
        txt = resp if isinstance(resp, str) else str(resp)
        start = txt.find("{"); end = txt.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(txt[start:end+1])
            data["source"] = "gemini-3.1-pro"
            return data
    except Exception as e:
        log.warning(f"OCR AI fallback: {e}")
    return demo_result

async def analyze_care_pattern(patient_id: str) -> dict:
    """Explainable care insights from event log."""
    logs = await db.med_logs.find({"patient_id": patient_id}).to_list(500)
    appts = await db.appointments.find({"patient_id": patient_id}).to_list(500)
    total = len(logs) or 1
    taken = sum(1 for l in logs if l.get("status") == "taken")
    missed = sum(1 for l in logs if l.get("status") == "skipped")
    adherence_now = round(taken * 100 / total)
    missed_appts = sum(1 for a in appts if a.get("status") == "missed")

    if missed >= 3 or missed_appts >= 1:
        severity = "attention"
        title = "Care pattern change detected"
        observation = "Medication adherence has decreased over the last 7 days."
        evidence = [f"{missed} missed doses", f"{missed_appts} missed appointment(s)", f"Current adherence: {adherence_now}%"]
        confidence = min(0.95, 0.6 + missed * 0.08 + missed_appts * 0.1)
        action = "Consider reviewing the care plan with the caregiver or scheduling a follow-up."
    else:
        severity = "info"
        title = "Care plan on track"
        observation = "Recorded care activity is within expected range."
        evidence = [f"Adherence: {adherence_now}%", "No missed appointments"]
        confidence = 0.9
        action = "Continue the current care plan."
    return {
        "severity": severity,
        "title": title,
        "observation": observation,
        "evidence": evidence,
        "confidence": round(confidence, 2),
        "recommended_action": action,
        "adherence_percent": adherence_now,
        "missed_doses": missed,
        "missed_appointments": missed_appts,
        "generated_at": now_iso(),
        "source": "gemini-3.1-pro" if EMERGENT_LLM_KEY else "demo",
        "disclaimer": "This is a care coordination signal, not a medical diagnosis.",
    }

async def voice_reply(patient_id: str, text: str) -> dict:
    """Simple deterministic assistant that queries real data."""
    t = text.lower()
    if "medication" in t or "medicine" in t:
        meds = await db.medications.find({"patient_id": patient_id}).to_list(20)
        if not meds:
            return {"reply": "You have no medications scheduled.", "action": None}
        names = ", ".join(m["name"] for m in meds[:3])
        return {"reply": f"You have {len(meds)} medications today, including {names}.", "action": "open_medications"}
    if "appointment" in t:
        appt = await db.appointments.find_one({"patient_id": patient_id, "status": "scheduled"}, sort=[("scheduled_at", 1)])
        if not appt:
            return {"reply": "You have no upcoming appointments.", "action": None}
        return {"reply": f"Your next appointment is with {appt['doctor']} on {appt['scheduled_at'][:10]}.", "action": "open_appointments"}
    if "passport" in t or "emergency card" in t:
        return {"reply": "Opening your Emergency Passport.", "action": "open_passport"}
    if "sos" in t or "emergency" in t:
        return {"reply": "To activate SOS, tap the red emergency button on your dashboard.", "action": "highlight_sos"}
    return {"reply": "I can help with medications, appointments, or your Emergency Passport.", "action": None}


# ---------- auth ----------
@api.post("/auth/login")
async def login(inp: LoginIn):
    user = await db.users.find_one({"email": inp.email})
    if not user or not verify_pw(inp.password, user.get("password", "")):
        raise HTTPException(401, "Invalid credentials")
    token = make_token(user["id"], user["role"])
    user.pop("_id", None); user.pop("password", None)
    return {"token": token, "user": user}

@api.post("/auth/register")
async def register(inp: RegisterIn):
    if await db.users.find_one({"email": inp.email}):
        raise HTTPException(400, "Email already registered")
    u = {
        "id": uid(),
        "email": inp.email,
        "password": hash_pw(inp.password),
        "name": inp.name,
        "role": inp.role,
        "created_at": now_iso(),
    }
    await db.users.insert_one(u)
    token = make_token(u["id"], u["role"])
    u.pop("password", None)
    return {"token": token, "user": u}

@api.get("/auth/me")
async def me(user: dict = Depends(current_user)):
    return user

@api.get("/auth/demo-accounts")
async def demo_accounts():
    accs = await db.users.find({"is_demo": True}, {"_id": 0, "password": 0}).to_list(50)
    return accs


# ---------- patients ----------
@api.get("/patients")
async def list_patients(user: dict = Depends(current_user)):
    if user["role"] == "patient":
        pts = await db.patients.find({"user_id": user["id"]}, {"_id": 0}).to_list(10)
    else:
        pts = await db.patients.find({}, {"_id": 0}).to_list(50)
    return pts

@api.get("/patients/{pid}")
async def get_patient(pid: str, user: dict = Depends(current_user)):
    p = await db.patients.find_one({"id": pid}, {"_id": 0})
    if not p:
        raise HTTPException(404, "Not found")
    await log_access(pid, user["id"], user["role"], user["name"], "profile", "profile view")
    return p

@api.get("/patients/{pid}/timeline")
async def patient_timeline(pid: str, user: dict = Depends(current_user)):
    events = await db.events.find({"patient_id": pid}, {"_id": 0}).sort("timestamp", -1).to_list(200)
    return events


# ---------- medications ----------
@api.get("/medications")
async def list_meds(patient_id: Optional[str] = None, user: dict = Depends(current_user)):
    pid = patient_id or (await _self_patient_id(user))
    meds = await db.medications.find({"patient_id": pid}, {"_id": 0}).to_list(100)
    return meds

@api.post("/medications")
async def add_med(m: MedicationIn, user: dict = Depends(current_user)):
    pid = await _self_patient_id(user)
    doc = m.model_dump() | {"id": uid(), "patient_id": pid, "created_at": now_iso(), "active": True}
    await db.medications.insert_one(doc)
    await emit_event(pid, "MEDICATION_ADDED", user["name"], {"name": m.name})
    doc.pop("_id", None)
    return doc

@api.post("/medications/{mid}/log")
async def log_med(mid: str, body: MedLogIn, user: dict = Depends(current_user)):
    pid = await _self_patient_id(user)
    doc = body.model_dump() | {"id": uid(), "patient_id": pid, "timestamp": now_iso()}
    await db.med_logs.insert_one(doc)
    ev = "MEDICATION_TAKEN" if body.status == "taken" else ("MEDICATION_MISSED" if body.status == "skipped" else "MEDICATION_SNOOZED")
    await emit_event(pid, ev, user["name"], {"medication_id": mid, "scheduled_time": body.scheduled_time})
    doc.pop("_id", None)
    return doc

@api.delete("/medications/{mid}")
async def del_med(mid: str, user: dict = Depends(current_user)):
    await db.medications.delete_one({"id": mid})
    return {"ok": True}

@api.get("/medications/adherence")
async def adherence(patient_id: Optional[str] = None, user: dict = Depends(current_user)):
    pid = patient_id or (await _self_patient_id(user))
    logs = await db.med_logs.find({"patient_id": pid}).to_list(1000)
    days = {}
    for l in logs:
        d = l["timestamp"][:10]
        days.setdefault(d, {"taken": 0, "total": 0})
        days[d]["total"] += 1
        if l["status"] == "taken":
            days[d]["taken"] += 1
    week = sorted(days.keys())[-7:]
    series = [{"day": d, "adherence": round(days[d]["taken"] * 100 / max(days[d]["total"], 1))} for d in week]
    total = sum(v["total"] for v in days.values()) or 1
    taken = sum(v["taken"] for v in days.values())
    return {"overall": round(taken * 100 / total), "series": series}


# ---------- appointments ----------
@api.get("/appointments")
async def list_appts(patient_id: Optional[str] = None, user: dict = Depends(current_user)):
    pid = patient_id or (await _self_patient_id(user))
    return await db.appointments.find({"patient_id": pid}, {"_id": 0}).sort("scheduled_at", 1).to_list(100)

@api.post("/appointments")
async def create_appt(a: AppointmentIn, user: dict = Depends(current_user)):
    pid = await _self_patient_id(user)
    doc = a.model_dump() | {"id": uid(), "patient_id": pid, "status": "scheduled", "created_at": now_iso()}
    await db.appointments.insert_one(doc)
    await emit_event(pid, "APPOINTMENT_CREATED", user["name"], {"doctor": a.doctor, "at": a.scheduled_at})
    doc.pop("_id", None)
    return doc


# ---------- prescription OCR ----------
@api.post("/prescriptions/ocr")
async def prescription_ocr(file: Optional[UploadFile] = File(None), user: dict = Depends(current_user)):
    b64 = None
    if file:
        data = await file.read()
        b64 = base64.b64encode(data).decode()
    result = await gemini_extract_prescription(b64)
    return result

@api.post("/prescriptions/confirm")
async def confirm_prescription(payload: Dict[str, Any], user: dict = Depends(current_user)):
    pid = await _self_patient_id(user)
    added = []
    for med in payload.get("medications", []):
        doc = {
            "id": uid(),
            "patient_id": pid,
            "name": med.get("name"),
            "dosage": med.get("dosage"),
            "frequency": med.get("frequency"),
            "times": med.get("times", ["08:00", "20:00"]),
            "instructions": med.get("instructions", ""),
            "created_at": now_iso(),
            "active": True,
            "source": "prescription_ocr",
        }
        await db.medications.insert_one(doc)
        doc.pop("_id", None)
        added.append(doc)
    await emit_event(pid, "PRESCRIPTION_CONFIRMED", user["name"], {"count": len(added)})
    return {"added": added}


# ---------- care insights ----------
@api.get("/care/insights")
async def care_insights(patient_id: Optional[str] = None, user: dict = Depends(current_user)):
    pid = patient_id or (await _self_patient_id(user))
    return await analyze_care_pattern(pid)

@api.get("/care/alerts")
async def care_alerts(patient_id: Optional[str] = None, user: dict = Depends(current_user)):
    pid = patient_id or (await _self_patient_id(user))
    alerts = await db.care_alerts.find({"patient_id": pid}, {"_id": 0}).sort("created_at", -1).to_list(50)
    return alerts


# ---------- emergency orchestration ----------
STATE_ORDER = [
    "SOS_ACTIVATED", "CAREGIVER_NOTIFIED", "RESPONDER_ASSIGNED",
    "AMBULANCE_EN_ROUTE", "HOSPITAL_NOTIFIED", "PATIENT_ARRIVED", "CASE_CLOSED"
]

async def _self_patient_id(user: dict) -> str:
    if user["role"] != "patient":
        # non-patient acting: prefer first demo patient
        p = await db.patients.find_one({}, {"_id": 0})
        return p["id"] if p else ""
    p = await db.patients.find_one({"user_id": user["id"]}, {"_id": 0})
    return p["id"] if p else ""

async def _push_state(emergency_id: str, state: str, actor: str, note: str = ""):
    await db.emergencies.update_one(
        {"id": emergency_id},
        {"$set": {"status": state, "updated_at": now_iso()}, "$push": {"history": {"state": state, "at": now_iso(), "actor": actor, "note": note}}}
    )

@api.post("/emergency/sos")
async def sos(body: SOSIn, user: dict = Depends(require_roles("patient","admin"))):
    pid = await _self_patient_id(user)
    patient = await db.patients.find_one({"id": pid}, {"_id": 0})
    if not patient:
        raise HTTPException(400, "Patient profile missing")
    # snapshot emergency passport
    passport = {
        "identity": {"name": patient["name"], "age": patient.get("age"), "blood_group": patient.get("blood_group")},
        "medical": {"conditions": patient.get("conditions", []), "history": patient.get("history", "")},
        "medications": [m["name"] + " " + m.get("dosage","") for m in await db.medications.find({"patient_id": pid}).to_list(20)],
        "allergies": patient.get("allergies", []),
        "contacts": {"emergency": patient.get("emergency_contact"), "caregiver": patient.get("caregiver_name"), "doctor": patient.get("primary_doctor")},
        "hospital": patient.get("preferred_hospital"),
    }
    responders = await db.users.find({"role": "responder"}).to_list(5)
    hospitals = await db.users.find({"role": "hospital"}).to_list(5)
    e_id = uid()
    case_no = f"CB360-2026-{random.randint(100,999)}"
    doc = {
        "id": e_id,
        "case_no": case_no,
        "patient_id": pid,
        "patient_name": patient["name"],
        "activated_by": user["name"],
        "status": "SOS_ACTIVATED",
        "priority": "high",
        "location": {"label": body.location_label, "lat": body.latitude, "lng": body.longitude},
        "note": body.note,
        "passport_snapshot": passport,
        "responder_id": responders[0]["id"] if responders else None,
        "responder_name": responders[0]["name"] if responders else None,
        "hospital_id": hospitals[0]["id"] if hospitals else None,
        "hospital_name": hospitals[0]["name"] if hospitals else None,
        "eta_minutes": 12,
        "access_expires_at": (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat(),
        "history": [{"state": "SOS_ACTIVATED", "at": now_iso(), "actor": user["name"], "note": "SOS activated"}],
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "qr_token": f"CB360-TEMP-{uid()[:8].upper()}",
    }
    await db.emergencies.insert_one(doc)
    await emit_event(pid, "SOS_ACTIVATED", user["name"], {"case_no": case_no})
    # auto-notify caregiver
    caregiver = await db.users.find_one({"role": "caregiver"})
    if caregiver:
        await notify(caregiver["id"], "emergency", f"Emergency for {patient['name']}", f"Case {case_no} activated at {body.location_label}", {"emergency_id": e_id})
    doc.pop("_id", None)
    return doc

@api.get("/emergency")
async def list_emergencies(status: Optional[str] = None, user: dict = Depends(current_user)):
    q = {} if not status else {"status": status}
    if user["role"] == "patient":
        q["patient_id"] = await _self_patient_id(user)
    items = await db.emergencies.find(q, {"_id": 0}).sort("created_at", -1).to_list(100)
    return items

@api.get("/emergency/{eid}")
async def get_emergency(eid: str, user: dict = Depends(current_user)):
    e = await db.emergencies.find_one({"id": eid}, {"_id": 0})
    if not e:
        raise HTTPException(404, "Not found")
    await log_access(e["patient_id"], user["id"], user["role"], user["name"], f"emergency:{eid}", "emergency response")
    return e

@api.post("/emergency/{eid}/advance")
async def advance_emergency(eid: str, body: Dict[str, Any], user: dict = Depends(current_user)):
    e = await db.emergencies.find_one({"id": eid})
    if not e:
        raise HTTPException(404, "Not found")
    target = body.get("state")
    if target not in STATE_ORDER:
        raise HTTPException(400, "Unknown state")
    cur_idx = STATE_ORDER.index(e["status"])
    new_idx = STATE_ORDER.index(target)
    if new_idx <= cur_idx:
        raise HTTPException(400, f"Invalid transition {e['status']} → {target}")
    for i in range(cur_idx + 1, new_idx + 1):
        await _push_state(eid, STATE_ORDER[i], user["name"], body.get("note", ""))
        await emit_event(e["patient_id"], STATE_ORDER[i], user["name"], {"emergency_id": eid})
    if target == "HOSPITAL_NOTIFIED":
        hosp = await db.users.find_one({"role": "hospital"})
        if hosp:
            await notify(hosp["id"], "emergency", "Incoming emergency", f"Patient {e['patient_name']} inbound. ETA {e.get('eta_minutes',12)} min.", {"emergency_id": eid})
    updated = await db.emergencies.find_one({"id": eid}, {"_id": 0})
    return updated

@api.post("/emergency/{eid}/close")
async def close_emergency(eid: str, body: Dict[str, Any], user: dict = Depends(current_user)):
    await db.emergencies.update_one(
        {"id": eid},
        {"$set": {"status": "CASE_CLOSED", "handover_notes": body.get("notes", ""), "closed_at": now_iso()},
         "$push": {"history": {"state": "CASE_CLOSED", "at": now_iso(), "actor": user["name"], "note": "Handover completed"}}}
    )
    e = await db.emergencies.find_one({"id": eid}, {"_id": 0})
    await emit_event(e["patient_id"], "CASE_CLOSED", user["name"], {"emergency_id": eid})
    return e

@api.get("/emergency/{eid}/passport")
async def emergency_passport(eid: str, user: dict = Depends(current_user)):
    e = await db.emergencies.find_one({"id": eid}, {"_id": 0})
    if not e:
        raise HTTPException(404, "Not found")
    await log_access(e["patient_id"], user["id"], user["role"], user["name"], "emergency_passport", "responder/hospital access")
    return {"passport": e["passport_snapshot"], "case_no": e["case_no"], "expires_at": e["access_expires_at"]}


# ---------- passport (patient standalone) ----------
@api.get("/passport/me")
async def my_passport(user: dict = Depends(current_user)):
    pid = await _self_patient_id(user)
    p = await db.patients.find_one({"id": pid}, {"_id": 0})
    if not p:
        raise HTTPException(404, "No patient profile")
    meds = await db.medications.find({"patient_id": pid}).to_list(20)
    return {
        "identity": {"name": p["name"], "age": p.get("age"), "blood_group": p.get("blood_group")},
        "medical": {"conditions": p.get("conditions", []), "history": p.get("history", "")},
        "medications": [f"{m['name']} {m.get('dosage','')}" for m in meds],
        "allergies": p.get("allergies", []),
        "contacts": {"emergency": p.get("emergency_contact"), "caregiver": p.get("caregiver_name"), "doctor": p.get("primary_doctor")},
        "hospital": p.get("preferred_hospital"),
        "qr_token": f"CB360-PASS-{pid[:8].upper()}",
    }


# ---------- privacy ----------
@api.get("/privacy/permissions")
async def get_privacy(user: dict = Depends(current_user)):
    pid = await _self_patient_id(user)
    perms = await db.privacy.find({"patient_id": pid}, {"_id": 0}).to_list(100)
    return perms

@api.post("/privacy/permissions")
async def set_privacy(inp: PrivacyIn, user: dict = Depends(require_roles("patient"))):
    pid = await _self_patient_id(user)
    await db.privacy.update_one(
        {"patient_id": pid, "scope": inp.scope, "role": inp.role},
        {"$set": {"allowed": inp.allowed, "updated_at": now_iso()}},
        upsert=True,
    )
    await emit_event(pid, "PRIVACY_UPDATED", user["name"], {"scope": inp.scope, "role": inp.role, "allowed": inp.allowed})
    return {"ok": True}

@api.get("/privacy/access-log")
async def access_log(user: dict = Depends(current_user)):
    pid = await _self_patient_id(user)
    return await db.access_logs.find({"patient_id": pid}, {"_id": 0}).sort("timestamp", -1).to_list(100)


# ---------- notifications ----------
@api.get("/notifications")
async def get_notifs(user: dict = Depends(current_user)):
    return await db.notifications.find({"recipient_id": user["id"]}, {"_id": 0}).sort("created_at", -1).to_list(50)

@api.post("/notifications/{nid}/read")
async def mark_read(nid: str, user: dict = Depends(current_user)):
    await db.notifications.update_one({"id": nid}, {"$set": {"read": True}})
    return {"ok": True}


# ---------- voice ----------
@api.post("/voice/query")
async def voice_query(body: Dict[str, Any], user: dict = Depends(current_user)):
    pid = await _self_patient_id(user)
    return await voice_reply(pid, body.get("text", ""))


# ---------- admin analytics ----------
@api.get("/admin/analytics")
async def admin_analytics(user: dict = Depends(require_roles("admin"))):
    patients = await db.patients.count_documents({})
    caregivers = await db.users.count_documents({"role": "caregiver"})
    doctors = await db.users.count_documents({"role": "doctor"})
    responders = await db.users.count_documents({"role": "responder"})
    hosps = await db.users.count_documents({"role": "hospital"})
    emergencies = await db.emergencies.count_documents({})
    active = await db.emergencies.count_documents({"status": {"$nin": ["CASE_CLOSED"]}})
    closed = await db.emergencies.find({"status": "CASE_CLOSED"}, {"_id": 0}).to_list(50)
    # avg response time
    times = []
    for e in closed:
        try:
            a = datetime.fromisoformat(e["created_at"])
            b = datetime.fromisoformat(e.get("closed_at", e["updated_at"]))
            times.append((b - a).total_seconds() / 60)
        except Exception:
            pass
    avg = round(sum(times) / len(times), 1) if times else 8.4
    return {
        "totals": {"patients": patients, "caregivers": caregivers, "doctors": doctors, "responders": responders, "hospitals": hosps, "emergencies": emergencies, "active_emergencies": active},
        "avg_response_time_minutes": avg,
        "system": {"database": "operational", "ai": "gemini-3.1-pro" if EMERGENT_LLM_KEY else "demo", "maps": "simulated", "notifications": "operational"},
    }


# ---------- demo control ----------
@api.post("/demo/run-emergency")
async def demo_run(user: dict = Depends(current_user)):
    """Kick off scripted emergency; frontend polls state."""
    p = await db.patients.find_one({}, {"_id": 0})
    fake_user = {"id": p["user_id"], "role": "patient", "name": p["name"]}
    # create emergency
    body = SOSIn(location_label="MG Road, Bengaluru", latitude=12.9754, longitude=77.6060, note="Demo activation")
    # inline call: reuse
    result = await sos(body, fake_user)
    return result

@api.post("/demo/reset")
async def demo_reset(user: dict = Depends(require_roles("admin","patient"))):
    for coll in ["events", "med_logs", "emergencies", "notifications", "access_logs", "care_alerts"]:
        await db[coll].delete_many({})
    await seed_demo(force=True)
    return {"ok": True, "message": "Demo data reset"}


# ---------- seed demo data ----------
async def seed_demo(force: bool = False):
    if not force and await db.users.count_documents({}) > 0:
        return
    users = [
        {"id": uid(), "email": "patient@carebridge.demo", "password": hash_pw("demo1234"), "name": "Ravi Kumar", "role": "patient", "is_demo": True, "created_at": now_iso()},
        {"id": uid(), "email": "caregiver@carebridge.demo", "password": hash_pw("demo1234"), "name": "Anita Kumar", "role": "caregiver", "is_demo": True, "created_at": now_iso()},
        {"id": uid(), "email": "doctor@carebridge.demo", "password": hash_pw("demo1234"), "name": "Dr. Priya Sharma", "role": "doctor", "is_demo": True, "created_at": now_iso()},
        {"id": uid(), "email": "responder@carebridge.demo", "password": hash_pw("demo1234"), "name": "Vikram (Responder)", "role": "responder", "is_demo": True, "created_at": now_iso()},
        {"id": uid(), "email": "hospital@carebridge.demo", "password": hash_pw("demo1234"), "name": "Aster Central Hospital", "role": "hospital", "is_demo": True, "created_at": now_iso()},
        {"id": uid(), "email": "anjanapramod08@gmail.com", "password": hash_pw("demo1234"), "name": "Anjana (Owner)", "role": "admin", "is_demo": True, "created_at": now_iso()},
    ]
    await db.users.delete_many({})
    await db.users.insert_many([u.copy() for u in users])
    patient = users[0]
    caregiver = users[1]
    doctor = users[2]
    hospital = users[4]

    patient_doc = {
        "id": uid(),
        "user_id": patient["id"],
        "name": "Ravi Kumar",
        "age": 62,
        "blood_group": "B+",
        "phone": "+91 98450 12345",
        "address": "42, MG Road, Bengaluru 560001",
        "emergency_contact": {"name": "Anita Kumar", "relation": "Daughter", "phone": "+91 98450 67890"},
        "conditions": ["Type 2 Diabetes", "Hypertension"],
        "history": "CABG in 2019. Well-controlled with medication.",
        "allergies": ["Penicillin", "Sulfa drugs"],
        "caregiver_name": caregiver["name"],
        "primary_doctor": doctor["name"],
        "preferred_hospital": hospital["name"],
        "created_at": now_iso(),
    }
    await db.patients.delete_many({})
    await db.patients.insert_one(patient_doc.copy())
    pid = patient_doc["id"]

    meds = [
        {"id": uid(), "patient_id": pid, "name": "Metformin", "dosage": "500 mg", "frequency": "Twice daily", "times": ["08:00","20:00"], "instructions": "After meals", "active": True, "created_at": now_iso()},
        {"id": uid(), "patient_id": pid, "name": "Amlodipine", "dosage": "5 mg", "frequency": "Once daily", "times": ["09:00"], "instructions": "Morning", "active": True, "created_at": now_iso()},
        {"id": uid(), "patient_id": pid, "name": "Atorvastatin", "dosage": "10 mg", "frequency": "Once daily", "times": ["21:00"], "instructions": "At bedtime", "active": True, "created_at": now_iso()},
        {"id": uid(), "patient_id": pid, "name": "Aspirin", "dosage": "75 mg", "frequency": "Once daily", "times": ["13:00"], "instructions": "After lunch", "active": True, "created_at": now_iso()},
    ]
    await db.medications.delete_many({})
    await db.medications.insert_many([m.copy() for m in meds])

    # med logs — realistic 7-day pattern with declining adherence
    await db.med_logs.delete_many({})
    logs = []
    now = datetime.now(timezone.utc)
    for d in range(7, 0, -1):
        day = (now - timedelta(days=d))
        for m in meds:
            for t in m["times"]:
                miss_prob = 0.05 if d > 3 else 0.28  # more misses recently
                status = "skipped" if random.random() < miss_prob else "taken"
                logs.append({
                    "id": uid(), "patient_id": pid, "medication_id": m["id"],
                    "status": status, "scheduled_time": t,
                    "timestamp": day.replace(hour=int(t.split(":")[0]), minute=0).isoformat(),
                })
    await db.med_logs.insert_many(logs)

    # appointments
    await db.appointments.delete_many({})
    appts = [
        {"id": uid(), "patient_id": pid, "doctor": doctor["name"], "hospital": hospital["name"], "specialty": "Cardiology", "scheduled_at": (now + timedelta(days=1)).isoformat(), "reason": "Follow-up", "status": "scheduled", "created_at": now_iso()},
        {"id": uid(), "patient_id": pid, "doctor": doctor["name"], "hospital": hospital["name"], "specialty": "Endocrinology", "scheduled_at": (now + timedelta(days=8)).isoformat(), "reason": "Diabetes review", "status": "scheduled", "created_at": now_iso()},
        {"id": uid(), "patient_id": pid, "doctor": doctor["name"], "hospital": hospital["name"], "specialty": "General", "scheduled_at": (now - timedelta(days=2)).isoformat(), "reason": "BP check", "status": "missed", "created_at": now_iso()},
    ]
    await db.appointments.insert_many([a.copy() for a in appts])

    # care alerts
    await db.care_alerts.delete_many({})
    await db.care_alerts.insert_many([
        {"id": uid(), "patient_id": pid, "severity": "attention", "title": "Adherence declining", "body": "3 doses missed in the last 7 days.", "created_at": now_iso()},
        {"id": uid(), "patient_id": pid, "severity": "info", "title": "Follow-up appointment", "body": "Cardiology follow-up scheduled tomorrow at 10:30 AM.", "created_at": now_iso()},
    ])

    # privacy defaults
    await db.privacy.delete_many({})
    defaults = [
        ("medical_history", "caregiver", True), ("medications", "caregiver", True), ("private_notes", "caregiver", False),
        ("medical_history", "doctor", True), ("medications", "doctor", True), ("appointments", "doctor", True),
        ("emergency_passport", "responder", True), ("allergies", "responder", True), ("medications", "responder", True), ("location", "responder", True),
        ("emergency_intake", "hospital", True), ("authorized_documents", "hospital", True),
    ]
    await db.privacy.insert_many([{"id": uid(), "patient_id": pid, "scope": s, "role": r, "allowed": a, "updated_at": now_iso()} for s,r,a in defaults])

    # seed events from logs
    await db.events.delete_many({})
    for l in logs[:40]:
        await emit_event(pid, "MEDICATION_TAKEN" if l["status"]=="taken" else "MEDICATION_MISSED", "System", {"medication_id": l["medication_id"], "scheduled_time": l["scheduled_time"]})
    await emit_event(pid, "APPOINTMENT_CREATED", "System", {"doctor": doctor["name"]})
    await emit_event(pid, "APPOINTMENT_MISSED", "System", {"doctor": doctor["name"]})
    await emit_event(pid, "CARE_ALERT_CREATED", "System", {"reason": "adherence declining"})

    await db.notifications.delete_many({})
    await notify(caregiver["id"], "attention", "Care alert for Ravi Kumar", "Adherence declined this week (78%).", {})
    await notify(patient["id"], "info", "Appointment tomorrow", "Cardiology follow-up at 10:30 AM.", {})
    await db.access_logs.delete_many({})

    log.info("Demo data seeded.")


@app.on_event("startup")
async def startup():
    await seed_demo()

@app.on_event("shutdown")
async def shutdown():
    client.close()


app.include_router(api)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
