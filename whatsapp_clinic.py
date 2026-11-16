import base64
import os
import sqlite3
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Annotated, Literal, Sequence, TypedDict
from zoneinfo import ZoneInfo

import redis
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field
from pywa import WhatsApp, types
from pywa.types import MessageType

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# Local tool imports (Make sure appointment_management.py and patient_complaint.py exist)
from appointment_management import book_appointment, check_appointment_status, check_doctor_availability
from patient_complaint import generate_ticket

# ---------------- Credentials ----------------
load_dotenv()
PHONENUMBER_ID = os.getenv("PHONENUMBER_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
APP_ID = os.getenv("APP_ID")
APP_SECRET = os.getenv("APP_SECRET")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
GOOGLE_GENERATIVE_AI = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLINIC_BASE_URL = os.getenv("CLINIC_BASE_URL")

openrouter_model = OpenAI(api_key=OPENAI_API_KEY, base_url="https://openrouter.ai/api/v1")

# Global WhatsApp instance reference
wa = None

def init_whatsapp(fastapi_app: FastAPI) -> WhatsApp:
    """Initializes and returns the WhatsApp client attached to the FastAPI app."""
    global wa
    wa = WhatsApp(
        phone_id=os.getenv("PHONENUMBER_ID"),
        token=os.getenv("ACCESS_TOKEN"),
        app_id=os.getenv("APP_ID"),
        app_secret=os.getenv("APP_SECRET"),
        server=fastapi_app,
        webhook_endpoint="/whatsapp/webhook",
        verify_token=os.getenv("VERIFY_TOKEN"),
    )
    
    # Register webhook handlers after initialization
    @wa.on_message()
    def handle_text_message(client: WhatsApp, message: types.Message):
        if message.type == MessageType.REACTION:
            return
        
        if is_duplicate_message(message.id):
            return

        incoming_text = extract_text_from_message(message)
        if not incoming_text:
            wa.send_message(to=message.from_user.wa_id, text="Sorry, I couldn't understand that message. Could you try again?")
            return

        try:
            message.mark_as_read()
        except Exception as e:
            print(f"mark_as_read failed (non-fatal): {e}")

        _executor.submit(process_message, message, incoming_text)

    return wa


def get_chat_model():
    return ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.5, api_key=GOOGLE_GENERATIVE_AI)


CLINIC_TIMEZONE = ZoneInfo("Asia/Karachi")

def get_current_date_context() -> str:
    now = datetime.now(CLINIC_TIMEZONE)
    context = f"Current Time: {now.strftime('%A, %Y-%m-%d %I:%M %p')}\n"
    context += "Upcoming 7-day calendar for booking reference:\n"
    for i in range(7):
        future_day = now + timedelta(days=i)
        context += f"- {future_day.strftime('%A')}: {future_day.strftime('%Y-%m-%d')}\n"
    return context.strip()


# ---------------- State ----------------
class BotState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    incoming_message: str
    wa_id: int
    classifier: Literal["appointment_management", "patient_query", "patient_complaint"]
    response_to_user: str
    flow_active: bool


# ---------------- Shared helpers ----------------
REROUTE_SENTINEL = "REROUTE"
REROUTE_INSTRUCTION = (
    "\n\nIMPORTANT: If the patient's latest message is clearly unrelated to your current task "
    f"(e.g. they ask something from a totally different topic), respond with EXACTLY the single "
    f"word {REROUTE_SENTINEL} and nothing else. Do not use this for anything else."
)

def extract_text(ai_message: AIMessage) -> str:
    content = ai_message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def sanitize_history_for_agent(history: list[BaseMessage]) -> list[BaseMessage]:
    cleaned: list[BaseMessage] = []
    skip_tool_results = False
    for msg in history:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            skip_tool_results = True
            continue
        if isinstance(msg, ToolMessage):
            if skip_tool_results:
                continue
        else:
            skip_tool_results = False
        cleaned.append(msg)
    return cleaned


# ---------------- Media -> text preprocessing ----------------
def extract_text_from_message(message: types.Message) -> str:
    try:
        if message.type == MessageType.TEXT:
            return message.text

        elif message.type == MessageType.AUDIO and message.audio.voice:
            print("Downloading voice note...")
            audio_bytes = message.audio.get_bytes()
            base64audio = base64.b64encode(audio_bytes).decode("utf-8")
            try:
                response = openrouter_model.chat.completions.create(
                    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Transcribe this audio precisely. Output ONLY the transcription text, nothing else.",
                                },
                                {"type": "input_audio", "input_audio": {"data": base64audio, "format": "ogg"}},
                            ],
                        }
                    ],
                    timeout=30,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"OpenRouter error in STT: {e}")
                return ""

        elif message.type == MessageType.IMAGE:
            image_bytes = message.image.get_bytes()
            mime_type = getattr(message.image, "mime_type", None) or "image/jpeg"
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            caption = message.image.caption or ""
            try:
                response = openrouter_model.chat.completions.create(
                    model="nvidia/nemotron-nano-12b-2-vl:free",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all text from this image or describe the image clearly so a search engine can index it.",
                                },
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                            ],
                        }
                    ],
                    timeout=30,
                )
                extracted_visual_text = response.choices[0].message.content or ""
                return f"Patient Caption: {caption} | Image content: {extracted_visual_text}".strip()
            except Exception as e:
                print(f"OpenRouter image processing error: {e}")
                return caption

    except Exception as e:
        print(f"Failed to download/process media: {e}")
        return ""

    return ""


# ---------------- Classifier ----------------
class IncomingMessageParser(BaseModel):
    category: Literal["appointment_management", "patient_query", "patient_complaint"] = Field(
        description="The category of incoming message."
    )


def classifier_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that classifies incoming WhatsApp messages sent to a clinic into one "
                "of the following categories: appointment_management, patient_query, or patient_complaint.\n\n"
                "appointment_management: the patient wants to BOOK, SCHEDULE, RESERVE, RESCHEDULE, or CANCEL an "
                "appointment, or check the status of an existing one.\n\n"
                "patient_query: general informational questions with no booking intent — doctor's qualifications, "
                "clinic timings, fees, services offered, location, parking, payment options.\n\n"
                "patient_complaint: a complaint or negative feedback about their experience.\n\n"
                "You should only respond with the category name, and nothing else.",
            ),
            ("human", "Classify the following message: {message}"),
        ]
    )


_BOOKING_INTENT_KEYWORDS = [
    "book", "booking", "appoitment", "appointent", "appoint",
    "appointment", "schedule", "reschedule", "cancel my", "slot",
    "reserve", "come at", "come in", "come today", "visit today",
    "i'll come", "ill come", "aana hai", "aa jau", "milna hai",
]


def looks_like_booking_intent(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _BOOKING_INTENT_KEYWORDS)


def structured_llm_model(chat_model: ChatGoogleGenerativeAI = None):
    chat_model = chat_model or get_chat_model()
    return chat_model.with_structured_output(IncomingMessageParser)


def classifier_agent(state: BotState):
    message = state["incoming_message"]

    if looks_like_booking_intent(message):
        return {"classifier": "appointment_management", "flow_active": True}

    prompt = classifier_prompt()
    structured_model = structured_llm_model()
    chain = prompt | structured_model
    response = chain.invoke({"message": message})
    return {"classifier": response.category, "flow_active": True}


# ---------------- Clinic Details loader ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CLINIC_DETAILS_PATH = os.path.join(BASE_DIR, "clinic_details.txt")

def load_clinic_details() -> str:
    if os.path.exists(_CLINIC_DETAILS_PATH):
        with open(_CLINIC_DETAILS_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "Halcyon Healthcare: Nazimabad No. 1, near UC Office and Aga Juice, Karachi. Dr. OBT - MEDICAL DIRECTOR (MBBS, MS). Timings: Mon-Sat 7:30 PM - 11:30 PM."

CLINIC_DETAILS_TEXT = load_clinic_details()


# ---------------- Agent: patient query ----------------
def patient_query_agent(state: BotState):
    chat_model = get_chat_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly, professional receptionist assistant for Halcyon Healthcare "
                "(Nazimabad No. 1, near UC Office and Aga Juice, Karachi). Answer patient questions about "
                "Dr. OBT - MEDICAL DIRECTOR, clinic timings, fees, services, parking, and payment options "
                "using the clinic details below:\n\n"
                f"{CLINIC_DETAILS_TEXT}\n\n"
                "If the patient sends a greeting, greet them back respectfully. Be precise and concise."
                + REROUTE_INSTRUCTION.replace(
                    "respond with EXACTLY the single word",
                    "— including if they want to BOOK, RESCHEDULE, or CANCEL an appointment — "
                    "respond with EXACTLY the single word",
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | chat_model

    history = sanitize_history_for_agent(state.get("messages", [])[:-1])
    current_user_message = state["incoming_message"]
    response = chain.invoke({"chat_history": history, "input": current_user_message})
    answer = extract_text(response).strip()

    if answer == REROUTE_SENTINEL:
        return {
            "messages": [],
            "response_to_user": "",
            "flow_active": False,
        }

    return {
        "messages": [response],
        "response_to_user": answer,
        "flow_active": False,
    }


# ---------------- Agent: appointment management ----------------
_APPOINTMENT_TOOLS = [check_doctor_availability, book_appointment, check_appointment_status]
_APPOINTMENT_TOOLS_BY_NAME = {t.name: t for t in _APPOINTMENT_TOOLS}


def appointment_management_agent(state: BotState):
    chat_model = get_chat_model().bind_tools(_APPOINTMENT_TOOLS)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the appointment booking assistant for Halcyon Healthcare over WhatsApp.
# CURRENT + NEXT 7 DAYS DATE WITH DAYS: {current_date}
# CLINIC FACTS:
* Doctor: Dr. OBT - MEDICAL DIRECTOR (MBBS, MS).
* Clinic hours: Monday - Saturday, 7:30 PM - 11:30 PM. Closed Sunday."""
                .replace("{current_date}", get_current_date_context())
                + REROUTE_INSTRUCTION,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    appointment_llm = prompt | chat_model

    history = sanitize_history_for_agent(state.get("messages", [])[:-1])
    current_user_message = state["incoming_message"]

    response = appointment_llm.invoke({"chat_history": history, "input": current_user_message})
    new_messages: list[BaseMessage] = [response]
    tool_results: list[str] = []

    while response.tool_calls:
        tool_messages = []
        for call in response.tool_calls:
            tool_fn = _APPOINTMENT_TOOLS_BY_NAME[call["name"]]
            tool_result = tool_fn.invoke(call["args"])
            tool_results.append(str(tool_result))
            tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))
        new_messages.extend(tool_messages)

        response = chat_model.invoke(history + [HumanMessage(content=current_user_message)] + new_messages)
        new_messages.append(response)

    answer = extract_text(response).strip()

    if answer == REROUTE_SENTINEL:
        return {
            "messages": [],
            "response_to_user": "",
            "flow_active": False,
        }

    task_complete = any(r.startswith("Appointment confirmed") for r in tool_results)

    return {
        "messages": new_messages,
        "response_to_user": answer,
        "flow_active": not task_complete,
    }


# ---------------- Agent: patient complaint ----------------
_COMPLAINT_TOOLS = [generate_ticket]
_COMPLAINT_TOOLS_BY_NAME = {t.name: t for t in _COMPLAINT_TOOLS}


def patient_complaint_agent(state: BotState):
    chat_model = get_chat_model().bind_tools(_COMPLAINT_TOOLS)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an experienced clinic patient-relations assistant handling patient complaints."
                + REROUTE_INSTRUCTION,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    ticket_llm = prompt | chat_model

    history = sanitize_history_for_agent(state.get("messages", [])[:-1])
    current_user_message = state["incoming_message"]

    response = ticket_llm.invoke({"chat_history": history, "input": current_user_message})
    new_messages: list[BaseMessage] = [response]
    tool_results: list[str] = []

    while response.tool_calls:
        tool_messages = []
        for call in response.tool_calls:
            tool_fn = _COMPLAINT_TOOLS_BY_NAME[call["name"]]
            tool_result = tool_fn.invoke(call["args"])
            tool_results.append(str(tool_result))
            tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))
        new_messages.extend(tool_messages)

        response = chat_model.invoke(history + [HumanMessage(content=current_user_message)] + new_messages)
        new_messages.append(response)

    answer = extract_text(response).strip()

    if answer == REROUTE_SENTINEL:
        return {
            "messages": [],
            "response_to_user": "",
            "flow_active": False,
        }

    task_complete = any("has been logged" in r for r in tool_results)

    return {
        "messages": new_messages,
        "response_to_user": answer,
        "flow_active": not task_complete,
    }


# ---------------- Routing ----------------
def entry_router(state: BotState) -> str:
    if state.get("flow_active") and state.get("classifier"):
        return f"{state['classifier']}_agent"
    return "classifier_agent"


def routing_agent(state: BotState) -> str:
    if state["classifier"] == "patient_complaint":
        return "patient_complaint_agent"
    if state["classifier"] == "patient_query":
        return "patient_query_agent"
    return "appointment_management_agent"


def after_agent_router(state: BotState) -> str:
    if state.get("response_to_user", "") == "":
        return "classifier_agent"
    return END


# ---------------- Build Graph ----------------
_conn = sqlite3.connect("conversations.db", check_same_thread=False)
checkpointer = SqliteSaver(_conn)

def build_graph():
    graph = StateGraph(BotState)
    graph.add_node("classifier_agent", classifier_agent)
    graph.add_node("patient_query_agent", patient_query_agent)
    graph.add_node("patient_complaint_agent", patient_complaint_agent)
    graph.add_node("appointment_management_agent", appointment_management_agent)

    graph.add_conditional_edges(START, entry_router)
    graph.add_conditional_edges("classifier_agent", routing_agent)
    graph.add_conditional_edges("patient_query_agent", after_agent_router)
    graph.add_conditional_edges("appointment_management_agent", after_agent_router)
    graph.add_conditional_edges("patient_complaint_agent", after_agent_router)

    return graph.compile(checkpointer=checkpointer)

compiled_graph = build_graph()

# ---------------- Background processing ----------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

_executor = ThreadPoolExecutor(max_workers=8)
_DEDUPE_TTL_SECONDS = 60 * 60 * 24
_LOCK_TIMEOUT_SECONDS = 120
_LOCK_BLOCKING_TIMEOUT_SECONDS = 130


def is_duplicate_message(message_id: str) -> bool:
    was_newly_set = redis_client.set(f"processed_msg:{message_id}", "1", nx=True, ex=_DEDUPE_TTL_SECONDS)
    return not was_newly_set


def get_user_lock(wa_id: str):
    return redis_client.lock(
        f"lock:user:{wa_id}",
        timeout=_LOCK_TIMEOUT_SECONDS,
        blocking_timeout=_LOCK_BLOCKING_TIMEOUT_SECONDS,
    )


def process_message(message: types.Message, incoming_text: str):
    wa_id = str(message.from_user.wa_id)
    lock = get_user_lock(wa_id)

    acquired = False
    try:
        acquired = lock.acquire(blocking=True)
    except Exception as e:
        print(f"Redis lock error for {wa_id}: {e}", flush=True)

    if not acquired:
        if wa:
            wa.send_message(to=message.from_user.wa_id, text="Sorry, I'm still processing your previous message. Please wait a moment.")
        return

    try:
        start = time.time()
        config = {"configurable": {"thread_id": wa_id}, "recursion_limit": 15}

        try:
            result = compiled_graph.invoke(
                {
                    "incoming_message": incoming_text,
                    "wa_id": wa_id,
                    "messages": [HumanMessage(content=incoming_text)],
                },
                config=config,
            )
            ai_reply = result.get("response_to_user") or "Sorry, I didn't quite catch that. Could you rephrase?"
        except Exception:
            traceback.print_exc()
            ai_reply = "Sorry, I'm having trouble right now. Please try again in a moment."
        
        print("AI REPLY: ", ai_reply)
        print(f"Processed in {time.time() - start:.2f}s", flush=True)
        
        try:
            if wa:
                wa.send_message(to=message.from_user.wa_id, text=ai_reply)
        except Exception as e:
            print(f"Failed to send message: {e}")
    finally:
        try:
            lock.release()
        except Exception as e:
            print(f"Non-fatal: lock release for {wa_id} failed: {e}", flush=True)