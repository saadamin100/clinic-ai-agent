import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr, Field, field_validator
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
from zoneinfo import ZoneInfo
import os
import sys
from dotenv import load_dotenv

# Global state for OPD control
branch_states = {}

def get_branch_state(branch: str):
    if branch not in branch_states:
        branch_states[branch] = {
            "is_opd_active": True,
            "opd_title": "General OPD"
        }
    return branch_states[branch]

class OPDToggleRequest(BaseModel):
    active: bool = True
    title: str = "Special OPD"
    
# 1. Sabse pehle load_dotenv run karo
load_dotenv()

# 2. Path setup
current_dir = os.path.dirname(os.path.abspath(__file__)) # yeh Backend folder hai
root_dir = os.path.abspath(os.path.join(current_dir, "..")) # yeh clinic_whatsapp folder hai
sys.path.insert(0, root_dir)

from fastapi import FastAPI

# 3. Main Central App initialize karo
app = FastAPI(title="Unified Clinic Central Backend")

# 4. whatsapp_clinic se sirf init_whatsapp import karo aur apni main app pass kar do
from whatsapp_clinic import init_whatsapp
init_whatsapp(app)

# 5. Baaki variables aur configurations
DB_PATH = "clinic_data.db"
DEFAULT_BRANCH = os.getenv("CLINIC_BRANCH", "Main Branch")
RETELL_API_KEY = os.getenv("RETELL_API_KEY")

# Agar init_whatsapp call karna hai toh yahan kar sakte hain:
# init_whatsapp(whatsapp_sub_app)
# Doctors + schedules
# --------------------------------------------------------------------------
DOCTORS = {
    "Dr. Muhammad Zubair Yousuf": {
        "specialty": "Dental Surgeon (BDS, RDS, MScDS, PDG-HP)",
        "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
        "start": "19:30",  # 7:30 PM
        "end": "23:30",    # 11:30 PM
        "slot_minutes": 30,
        "fee_pkr": 500,
    },
}

WEEKDAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
VALID_STATUSES = [
    "Scheduled",
    "Waiting",
    "Checked In",
    "In Consultation",
    "Completed",
    "Cancelled",
    "No-Show",
]


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id INTEGER PRIMARY KEY,
                patient_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                doctor_name TEXT NOT NULL,
                appt_date TEXT NOT NULL,   -- 'YYYY-MM-DD'
                appt_time TEXT NOT NULL,   -- 'HH:MM'
                status TEXT NOT NULL,
                fee_pkr INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                token_id INTEGER,
                type TEXT NOT NULL DEFAULT 'ONLINE',
                branch TEXT NOT NULL DEFAULT 'Main Branch',
                reminder_sent INTEGER DEFAULT 0,
                alert_sent INTEGER DEFAULT 0,
                token_reminder_sent INTEGER DEFAULT 0
            )
            """
        )
        for column, definition in (
            ("token_id", "INTEGER"),
            ("type", "TEXT NOT NULL DEFAULT 'ONLINE'"),
            ("branch", "TEXT NOT NULL DEFAULT 'Main Branch'"),
            ("reminder_sent", "INTEGER DEFAULT 0"),
            ("alert_sent", "INTEGER DEFAULT 0"),
            ("token_reminder_sent", "INTEGER DEFAULT 0"),
        ):
            try:
                cur.execute(f"ALTER TABLE appointments ADD COLUMN {column} {definition}")
            except sqlite3.OperationalError as error:
                if "duplicate column name" not in str(error).lower():
                    raise
        cur.execute(
            "UPDATE appointments SET token_id = appointment_id WHERE token_id IS NULL"
        )
        cur.execute(
            "UPDATE appointments SET branch = ? WHERE branch IS NULL OR branch = ''",
            (DEFAULT_BRANCH,),
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        cur.execute("SELECT value FROM meta WHERE key = 'next_appointment_id'")
        if cur.fetchone() is None:
            cur.execute("INSERT INTO meta (key, value) VALUES ('next_appointment_id', '1001')")
        cur.execute("SELECT value FROM meta WHERE key = 'next_token_id'")
        if cur.fetchone() is None:
            cur.execute("SELECT COALESCE(MAX(token_id), 1000) AS max_token FROM appointments")
            next_token = int(cur.fetchone()["max_token"]) + 1
            cur.execute(
                "INSERT INTO meta (key, value) VALUES ('next_token_id', ?)",
                (str(max(next_token, 1001)),),
            )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                ticket_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                complaint TEXT NOT NULL,
                date_time TEXT NOT NULL
            )
            """
        )

def get_next_appointment_id(cur) -> int:
    cur.execute("SELECT value FROM meta WHERE key = 'next_appointment_id'")
    next_id = int(cur.fetchone()["value"])
    cur.execute("UPDATE meta SET value = ? WHERE key = 'next_appointment_id'", (str(next_id + 1),))
    return next_id


def get_next_token(cur) -> int:
    cur.execute("SELECT value FROM meta WHERE key = 'next_token_id'")
    token_id = int(cur.fetchone()["value"])
    cur.execute(
        "UPDATE meta SET value = ? WHERE key = 'next_token_id'",
        (str(token_id + 1),),
    )
    return token_id


# --------------------------------------------------------------------------
# Slot generation
# --------------------------------------------------------------------------
def generate_slots(doctor: dict, target_date: date_cls) -> list[str]:
    weekday = WEEKDAY_ABBR[target_date.weekday()]
    if weekday not in doctor["days"]:
        return []

    start_h, start_m = map(int, doctor["start"].split(":"))
    end_h, end_m = map(int, doctor["end"].split(":"))
    slot_minutes = doctor["slot_minutes"]

    slots = []
    current = datetime.combine(target_date, datetime.min.time()).replace(hour=start_h, minute=start_m)
    end = datetime.combine(target_date, datetime.min.time()).replace(hour=end_h, minute=end_m)

    while current + timedelta(minutes=slot_minutes) <= end:
        slots.append(current.strftime("%H:%M"))
        current += timedelta(minutes=slot_minutes)

    return slots


def get_available_slots(
    doctor_name: str,
    target_date: date_cls,
    branch: str = DEFAULT_BRANCH,
) -> list[str]:
    doctor = DOCTORS[doctor_name]
    all_slots = generate_slots(doctor, target_date)
    if not all_slots:
        return []

    with get_conn() as conn:
        booked = conn.execute(
            """SELECT appt_time FROM appointments
               WHERE doctor_name = ? AND appt_date = ? AND branch = ?
                 AND status NOT IN ('Cancelled', 'No-Show')""",
            (doctor_name, target_date.isoformat(), branch),
        ).fetchall()
    booked_times = {row["appt_time"] for row in booked}

    return [s for s in all_slots if s not in booked_times]


# --------------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------------
class BookAppointmentRequest(BaseModel):
    patient_name: str = Field(..., min_length=1)
    phone: str = Field(..., min_length=5)
    doctor_name: str = "Dr. Muhammad Zubair Yousuf"
    appt_date: str = Field(..., description="Date in YYYY-MM-DD format")
    appt_time: str = Field(..., description="Time in HH:MM 24h format")
    branch: str = DEFAULT_BRANCH

    @field_validator("doctor_name")
    @classmethod
    def doctor_must_exist(cls, v):
        if v not in DOCTORS:
            raise ValueError(f"Unknown doctor '{v}'. Valid doctors: {list(DOCTORS.keys())}")
        return v

    @field_validator("patient_name")
    @classmethod
    def name_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("patient_name must not be blank")
        return v.strip()

    @field_validator("appt_date")
    @classmethod
    def date_must_be_valid(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("appt_date must be in YYYY-MM-DD format")
        return v

    @field_validator("appt_time")
    @classmethod
    def time_must_be_valid(cls, v):
        try:
            datetime.strptime(v, "%H:%M")
        except ValueError:
            raise ValueError("appt_time must be in HH:MM 24h format")
        return v


class AppointmentResponse(BaseModel):
    appointment_id: int
    patient_name: str
    phone: str
    doctor_name: str
    specialty: str
    appt_date: str
    appt_time: str
    status: str
    fee_pkr: int
    token_id: int | None = None
    type: str = "ONLINE"
    branch: str = DEFAULT_BRANCH


class AvailabilityResponse(BaseModel):
    doctor_name: str
    specialty: str
    date: str
    available_slots: list[str]


class WalkInRequest(BaseModel):
    patient_name: str = Field(..., min_length=1)
    phone: str = Field(..., min_length=5)
    doctor_name: str = next(iter(DOCTORS))
    branch: str = DEFAULT_BRANCH
    fee_pkr: int | None = None


# --------------------------------------------------------------------------
# WhatsApp Notification Helpers
# --------------------------------------------------------------------------
import os
import httpx

async def send_whatsapp_message(phone: str, message: str):
    print("--- [DEBUG] send_whatsapp_message function CALL HOGYA HAI ---")
    
    whatsapp_token = os.getenv("ACCESS_TOKEN")
    phone_number_id = os.getenv("PHONENUMBER_ID")

    print(f"[DEBUG] Token loaded? {bool(whatsapp_token)} | Phone ID loaded? {bool(phone_number_id)}")

    if not whatsapp_token or not phone_number_id:
        print("[WHATSAPP ERROR] WHATSAPP_TOKEN ya WHATSAPP_PHONE_NUMBER_ID .env file mein nahi mil rahay!")
        return

    # Phone number se non-digit characters remove karein
    clean_phone = "".join(filter(str.isdigit, phone))
    
    # Agar number '0' se shuru ho raha hai (e.g. 0304...), toh shuru ka 0 hata kar '92' laga dein
    if clean_phone.startswith("0"):
        clean_phone = "92" + clean_phone[1:]
    elif not clean_phone.startswith("92") and len(clean_phone) == 10:
        clean_phone = "92" + clean_phone

    print(f"[DEBUG] Cleaned & Formatted phone number: {clean_phone}")
    
    url = f"https://graph.facebook.com/v17.0/{phone_number_id}/messages"
    
    headers = {
        "Authorization": f"Bearer {whatsapp_token}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": clean_phone,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message
        }
    }

    try:
        print("[DEBUG] Meta API ko request bhej rahay hain...")
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=15.0)
            print(f"[DEBUG] Meta API response status code: {response.status_code}")
            print(f"[DEBUG] Meta API response body: {response.text}")
            
            if response.status_code == 200:
                print(f"[WHATSAPP SENT] Successfully sent to {clean_phone}")
            else:
                print(f"[WHATSAPP ERROR] Failed! Status: {response.status_code}")
    except Exception as e:
        print(f"[WHATSAPP EXCEPTION] Error aagya connection mein: {str(e)}")
                
async def check_online_appointment_reminders():
    with get_conn() as conn:
        now = datetime.now()
        today_str = now.date().isoformat()
        
        # 2 Hours Before Reminder
        target_time_start = (now + timedelta(hours=1, minutes=55)).strftime("%H:%M")
        target_time_end = (now + timedelta(hours=2, minutes=5)).strftime("%H:%M")

        rows = conn.execute(
            """SELECT * FROM appointments 
               WHERE appt_date = ? AND type = 'ONLINE' 
               AND status = 'Scheduled' AND reminder_sent = 0
               AND appt_time BETWEEN ? AND ?""",
            (today_str, target_time_start, target_time_end)
        ).fetchall()

        for row in rows:
            msg = f"Reminder: {row['patient_name']} ji, aapki appointment aaj {row['appt_time']} par scheduled hai. Baraye meherbani waqt par tashreef layein."
            await send_whatsapp_message(row["phone"], msg)
            conn.execute(
                "UPDATE appointments SET reminder_sent = 1 WHERE appointment_id = ?",
                (row["appointment_id"],)
            )


# App Initialization & Scheduler

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def on_startup():
    init_db()
    scheduler.add_job(check_online_appointment_reminders, 'interval', minutes=5)
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()


def appointment_response(row) -> AppointmentResponse:
    return AppointmentResponse(
        appointment_id=row["appointment_id"],
        patient_name=row["patient_name"],
        phone=row["phone"],
        doctor_name=row["doctor_name"],
        specialty=DOCTORS.get(row["doctor_name"], {}).get("specialty", "Unknown"),
        appt_date=row["appt_date"],
        appt_time=row["appt_time"],
        status=row["status"],
        fee_pkr=row["fee_pkr"],
        token_id=row["token_id"],
        type=row["type"] or "ONLINE",
        branch=row["branch"] or DEFAULT_BRANCH,
    )


def dashboard_page(filename: str) -> FileResponse:
    path = os.path.join(os.path.dirname(__file__), filename)
    return FileResponse(path, media_type="text/html")


# ---- Retell AI Webhook ----------------------------------------------------
@app.post("/api/retell-webhook", tags=["Retell AI"])
async def retell_webhook(request: Request):
    body = await request.json() 
    print("Retell Webhook Payload:", json.dumps(body, indent=2))   
    interaction_type = body.get("interaction_type")    
    
    if interaction_type == "call_tool_call" or "name" in body or "args" in body:
        function_name = body.get("name") or body.get("tool", {}).get("name")
        args = body.get("args") or body.get("tool_call", {}).get("args", {})
        
        if not args:
            args = body          
             
        if function_name == "book_appointment" or "name" in args or "phone" in args:
            patient_name = args.get("name") or args.get("patient_name")
            phone = args.get("phone")
            date_str = args.get("date") or args.get("appt_date")
            time_str = args.get("time") or args.get("appt_time")
            doctor_name = "Dr. Muhammad Zubair Yousuf"

            if not all([patient_name, phone, date_str, time_str]):
                return JSONResponse(content={
                    "result": "Please provide all required details: name, phone number, date, and time to book the appointment."
                })

            try:
                date_lower = str(date_str).strip().lower()
                now_utc = datetime.now(timezone.utc)
                if "today" in date_lower:
                    target_date = now_utc.date()
                elif "tomorrow" in date_lower:
                    target_date = (now_utc + timedelta(days=1)).date()
                else:
                    try:
                        target_date = datetime.strptime(str(date_str).strip(), "%Y-%m-%d").date()
                    except ValueError:
                        target_date = now_utc.date()

                date_formatted = target_date.isoformat()

                time_clean = str(time_str).strip().upper()
                try:
                    parsed_time = datetime.strptime(time_clean, "%H:%M")
                    formatted_time = parsed_time.strftime("%H:%M")
                except ValueError:
                    try:
                        parsed_time = datetime.strptime(time_clean, "%I:%M %p")
                        formatted_time = parsed_time.strftime("%H:%M")
                    except ValueError:
                        formatted_time = time_clean

                doctor = DOCTORS[doctor_name]
                
                available_slots = get_available_slots(doctor_name, target_date, DEFAULT_BRANCH)
                if formatted_time not in available_slots and available_slots:
                    slots_str = ", ".join(available_slots[:5])
                    return JSONResponse(content={
                        "result": f"Sorry, the time {formatted_time} is not available. Available slots are: {slots_str}."
                    })

                now_iso = now_utc.isoformat()
                
                with get_conn() as conn:
                    cur = conn.cursor()
                    new_id = get_next_appointment_id(cur)
                    token_id = get_next_token(cur)
                    cur.execute(
                        """INSERT INTO appointments
                            (appointment_id, patient_name, phone, doctor_name, appt_date,
                             appt_time, status, fee_pkr, created_at, token_id, type, branch, reminder_sent, alert_sent, token_reminder_sent)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0)""",
                        (
                            new_id,
                            patient_name,
                            phone,
                            doctor_name,
                            date_formatted,
                            formatted_time,
                            "Scheduled",
                            doctor["fee_pkr"],
                            now_iso,
                            token_id,
                            "ONLINE",
                            DEFAULT_BRANCH,
                        ),
                    )
                    conn.commit()

                try:
                    msg = f"Hello {patient_name}, your appointment is confirmed with Dr. Muhammad Zubair Yousuf on {date_formatted} at {formatted_time}. Token ID: #{token_id}. Thank you!"
                    await send_whatsapp_message(phone, msg)
                except Exception as wa_err:
                    print("WhatsApp Notification Error:", wa_err)

                return JSONResponse(content={
                    "result": f"Appointment successfully booked for {patient_name} on {date_formatted} at {formatted_time} with {doctor_name}. Your Token ID is #{token_id}."
                })

            except Exception as e:
                print(f"Retell Booking Error: {str(e)}")
                return JSONResponse(content={
                    "result": f"Failed to book appointment due to a system error: {str(e)}"
                }, status_code=500)
                 
    return JSONResponse(content={"status": "received"})


# ---- Doctors -----------------------------------------------------------
@app.get("/api/doctors", tags=["Doctors"])
def list_doctors():
    return {
        name: {
            "specialty": d["specialty"],
            "days": d["days"],
            "hours": f"{d['start']} - {d['end']}",
            "fee_pkr": d["fee_pkr"],
        }
        for name, d in DOCTORS.items()
    }


# ---- Availability --------------------------------------------------------
@app.get("/api/availability/{doctor_name}", response_model=AvailabilityResponse, tags=["Appointments"])
def check_availability(
    doctor_name: str,
    date: str,
    branch: str = DEFAULT_BRANCH,
):
    if doctor_name not in DOCTORS:
        raise HTTPException(status_code=404, detail=f"'{doctor_name}' is not a doctor at this clinic")

    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail="date must be in YYYY-MM-DD format")

    if target_date < datetime.now(timezone.utc).date():
        raise HTTPException(status_code=400, detail="Cannot check availability for a past date")

    slots = get_available_slots(doctor_name, target_date, branch)
    return AvailabilityResponse(
        doctor_name=doctor_name,
        specialty=DOCTORS[doctor_name]["specialty"],
        date=date,
        available_slots=slots,
    )


# ---- Book appointment ------------------------------------------------------
@app.post(
    "/api/appointments/book",
    response_model=AppointmentResponse,
    status_code=201,
    tags=["Appointments"],
)
def book_appointment(payload: BookAppointmentRequest):
    target_date = datetime.strptime(payload.appt_date, "%Y-%m-%d").date()
    if target_date < datetime.now(timezone.utc).date():
        raise HTTPException(status_code=400, detail="Cannot book an appointment in the past")

    available_slots = get_available_slots(
        payload.doctor_name,
        target_date,
        payload.branch,
    )
    if payload.appt_time not in available_slots:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"That slot is not available for {payload.doctor_name} on {payload.appt_date}",
                "available_slots": available_slots,
            },
        )
    doctor = DOCTORS[payload.doctor_name]
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        cur = conn.cursor()
        new_id = get_next_appointment_id(cur)
        token_id = get_next_token(cur)
        cur.execute(
            """INSERT INTO appointments
               (appointment_id, patient_name, phone, doctor_name, appt_date,
                appt_time, status, fee_pkr, created_at, token_id, type, branch, reminder_sent, alert_sent, token_reminder_sent)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0)""",
            (
                new_id,
                payload.patient_name,
                payload.phone,
                payload.doctor_name,
                payload.appt_date,
                payload.appt_time,
                "Scheduled",
                doctor["fee_pkr"],
                now,
                token_id,
                "ONLINE",
                payload.branch,
            ),
        )
    return AppointmentResponse(
        appointment_id=new_id,
        patient_name=payload.patient_name,
        phone=payload.phone,
        doctor_name=payload.doctor_name,
        specialty=doctor["specialty"],
        appt_date=payload.appt_date,
        appt_time=payload.appt_time,
        status="Scheduled",
        fee_pkr=doctor["fee_pkr"],
        token_id=token_id,
        type="ONLINE",
        branch=payload.branch,
    )


@app.get("/api/appointments/today", tags=["Dashboard"])
def today_online_appointments(branch: str = DEFAULT_BRANCH, date: Optional[str] = None):
    date_to_use = date if date else datetime.now(timezone.utc).date().isoformat()
    
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM appointments
                WHERE DATE(appt_date) = ? AND branch = ? AND UPPER(type) = 'ONLINE'
                ORDER BY appt_time, token_id""",
            (date_to_use, branch),
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/appointments/{appointment_id}", response_model=AppointmentResponse, tags=["Appointments"])
def get_appointment(appointment_id: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Appointment {appointment_id} not found")
    return appointment_response(row)


@app.post("/api/appointments/{appointment_id}/cancel", response_model=AppointmentResponse, tags=["Appointments"])
def cancel_appointment(appointment_id: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Appointment {appointment_id} not found")
        if row["status"] != "Scheduled":
            raise HTTPException(status_code=400, detail=f"Appointment is already '{row['status']}', cannot cancel")

        conn.execute(
            "UPDATE appointments SET status = 'Cancelled' WHERE appointment_id = ?", (appointment_id,)
        )
    return AppointmentResponse(
        appointment_id=row["appointment_id"],
        patient_name=row["patient_name"],
        phone=row["phone"],
        doctor_name=row["doctor_name"],
        specialty=DOCTORS.get(row["doctor_name"], {}).get("specialty", "Unknown"),
        appt_date=row["appt_date"],
        appt_time=row["appt_time"],
        status="Cancelled",
        fee_pkr=row["fee_pkr"],
    )


@app.get("/api/appointments", tags=["Appointments"])
def list_appointments():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM appointments ORDER BY appt_date, appt_time").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/branches", tags=["Dashboard"])
def list_branches():
    return {"branches": [DEFAULT_BRANCH], "default_branch": DEFAULT_BRANCH}


@app.get("/api/display/queue", tags=["Dashboard"])
def display_queue(branch: str = DEFAULT_BRANCH):
    today = datetime.now(ZoneInfo("Asia/Karachi")).date().isoformat()
    with get_conn() as conn:
        now_serving = conn.execute(
            """SELECT token_id, patient_name, phone, type, status, appt_time FROM appointments
               WHERE DATE(appt_date) = ? AND branch = ? AND status = 'In Consultation'
               ORDER BY token_id DESC LIMIT 1""",
            (today, branch),
        ).fetchone()
        rows = conn.execute(
            """SELECT token_id, patient_name, type, status, appt_time
               FROM appointments
               WHERE DATE(appt_date) = ? AND branch = ? AND status IN ('Scheduled', 'Waiting', 'Checked In')
               ORDER BY CASE WHEN type = 'ONLINE' THEN 0 ELSE 1 END,
                        appt_time, token_id""",
            (today, branch),
        ).fetchall()
        
        state = get_branch_state(branch)
        
    return {
        "branch": branch,
        "now_serving": dict(now_serving) if now_serving else None,
        "waiting_queue": [dict(row) for row in rows],
        "is_opd_active": state["is_opd_active"],  # <-- Yeh frontend ko batayega ke OPD on hai ya off
        "opd_title": state["opd_title"]
    }
    
@app.post("/api/queue/add_walkin", tags=["Dashboard"])
async def add_walkin(payload: WalkInRequest):
    karachi_tz = ZoneInfo("Asia/Karachi")
    now_local = datetime.now(karachi_tz)
    today = now_local.date().isoformat()
    now_time = now_local.strftime("%H:%M")
    
    doctor = DOCTORS[payload.doctor_name]
    with get_conn() as conn:
        conflict = conn.execute(
            """SELECT token_id FROM appointments
               WHERE appt_date = ? AND appt_time = ? AND branch = ?
                 AND type = 'ONLINE' AND status NOT IN ('Completed', 'Cancelled', 'No-Show')
               LIMIT 1""",
            (today, now_time, payload.branch),
        ).fetchone()
        if conflict:
            raise HTTPException(
                status_code=409,
                detail=f"Online appointment token #{conflict['token_id']} already occupies {now_time}.",
            )
        appointment_id = get_next_appointment_id(conn.cursor())
        token_id = get_next_token(conn.cursor())
        conn.execute(
            """INSERT INTO appointments
               (appointment_id, patient_name, phone, doctor_name, appt_date, appt_time,
                status, fee_pkr, created_at, token_id, type, branch, reminder_sent, alert_sent, token_reminder_sent)
               VALUES (?, ?, ?, ?, ?, ?, 'Waiting', ?, ?, ?, 'WALK_IN', ?, 0, 0, 0)""",
            (
                appointment_id,
                payload.patient_name,
                payload.phone,
                payload.doctor_name,
                today,
                now_time,
                payload.fee_pkr if payload.fee_pkr is not None else doctor["fee_pkr"],
                now_local.isoformat(),
                token_id,
                payload.branch,
            ),
        )
        conn.commit()
        
        row = conn.execute(
            "SELECT * FROM appointments WHERE appointment_id = ?",
            (appointment_id,),
        ).fetchone()

    # --- Trigger WhatsApp Alert for Walk-In ---
    try:
        msg = f"Hello {payload.patient_name}, your walk-in appointment is registered with {payload.doctor_name} for today at {now_time}. Token ID: #{token_id}. Status: Waiting. Thank you!"
        await send_whatsapp_message(payload.phone, msg)
    except Exception as wa_err:
        print("Walk-In WhatsApp Error:", wa_err)

    return dict(row)


@app.post("/api/appointments/checkin/{appointment_id}", tags=["Dashboard"])
def checkin_appointment(appointment_id: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM appointments WHERE appointment_id = ?",
            (appointment_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Appointment not found")
        conn.execute(
            "UPDATE appointments SET status = 'Checked In' WHERE appointment_id = ?",
            (appointment_id,),
        )
    return {"status": "success", "token_id": row["token_id"]}


@app.post("/api/queue/call-next", tags=["Dashboard"])
async def next_patient(branch: str = DEFAULT_BRANCH):
    today = datetime.now(ZoneInfo("Asia/Karachi")).date().isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE appointments SET status = 'Completed' WHERE DATE(appt_date) = ? AND branch = ? AND status = 'In Consultation'",
            (today, branch),
        )
        
        row = conn.execute(
            """SELECT * FROM appointments
               WHERE DATE(appt_date) = ? AND branch = ?
                 AND status IN ('Scheduled', 'Waiting', 'Checked In')
               ORDER BY CASE WHEN type = 'ONLINE' THEN 0 ELSE 1 END,
                        appt_time, token_id LIMIT 1""",
            (today, branch),
        ).fetchone()
        
        if row is None:
            return {"status": "empty", "branch": branch}
             
        conn.execute(
            "UPDATE appointments SET status = 'In Consultation' WHERE appointment_id = ?",
            (row["appointment_id"],),
        )
        
        current_serving_token = row["token_id"]

    # 1. Walk-in Alert (3 tokens before)
    target_walkin_token = current_serving_token + 3 if current_serving_token else 0
    
    with get_conn() as conn:
        walkin_upcoming = conn.execute(
            """SELECT * FROM appointments 
               WHERE appt_date = ? AND branch = ? AND type = 'WALK_IN' 
               AND token_id = ? AND alert_sent = 0""",
            (today, branch, target_walkin_token)
        ).fetchone()

        if walkin_upcoming:
            msg = f"Alert: {walkin_upcoming['patient_name']} ji, aapka token number #{target_walkin_token} hai. Ab se 3 tokens baad aapki baari aane wali hai, baraye meherbani clinic mein tashreef rakhein."
            await send_whatsapp_message(walkin_upcoming["phone"], msg)
            conn.execute(
                "UPDATE appointments SET alert_sent = 1 WHERE appointment_id = ?",
                (walkin_upcoming["appointment_id"],)
            )

    # 2. Online Reminder (2 tokens before)
    target_online_token = current_serving_token + 2 if current_serving_token else 0
    
    with get_conn() as conn:
        online_upcoming = conn.execute(
            """SELECT * FROM appointments 
               WHERE appt_date = ? AND branch = ? AND type = 'ONLINE' 
               AND token_id = ? AND token_reminder_sent = 0""",
            (today, branch, target_online_token)
        ).fetchone()

        if online_upcoming:
            msg = f"Reminder: {online_upcoming['patient_name']} ji, aapka online token #{target_online_token} hai. Ab se sirf 2 tokens baqi hain, baraye meherbani waqt par tashreef rakhein."
            await send_whatsapp_message(online_upcoming["phone"], msg)
            conn.execute(
                "UPDATE appointments SET token_reminder_sent = 1 WHERE appointment_id = ?",
                (online_upcoming["appointment_id"],)
            )

    return {"status": "success", "token_id": row["token_id"], "patient_name": row["patient_name"]}


@app.post("/api/doctor/opd-toggle", tags=["Doctor OPD"])
def toggle_opd(branch: str = DEFAULT_BRANCH, active: bool = True, title: str = "Special OPD"):
    state = get_branch_state(branch)
    state["is_opd_active"] = active
    state["opd_title"] = title
    return {
        "status": "success",
        "branch": branch,
        "is_opd_active": state["is_opd_active"],
        "message": f"OPD is now {'ACTIVE' if active else 'FINISHED'}"
    }
    
@app.get("/api/dashboard/summary", tags=["Dashboard"])
def dashboard_summary(branch: str = DEFAULT_BRANCH):
    today = datetime.now(ZoneInfo("Asia/Karachi")).date().isoformat()
    with get_conn() as conn:
        row = conn.execute(
            """SELECT
                  SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) AS completed,
                  COALESCE(SUM(CASE WHEN status = 'Completed' THEN fee_pkr ELSE 0 END), 0) AS revenue,
                  SUM(CASE WHEN status IN ('Scheduled', 'Waiting', 'Checked In') THEN 1 ELSE 0 END) AS waiting
               FROM appointments WHERE DATE(appt_date) = ? AND branch = ?""",
            (today, branch),
        ).fetchone()
    return {
        "completed_patients": row["completed"] or 0,
        "total_earnings": row["revenue"] or 0,
        "waiting_count": row["waiting"] or 0,
    }


@app.get("/receptionist", include_in_schema=False)
@app.get("/receptionist.html", include_in_schema=False)
def receptionist_dashboard():
    return dashboard_page("receptionist.html")


@app.get("/doctor", include_in_schema=False)
@app.get("/doctor.html", include_in_schema=False)
def doctor_dashboard():
    return dashboard_page("doctor.html")


@app.get("/display", include_in_schema=False)
@app.get("/display.html", include_in_schema=False)
def waiting_display():
    return waiting_display("display.html")


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Clinic Appointment API", "docs": "/docs"}


@app.exception_handler(ValueError)
def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
