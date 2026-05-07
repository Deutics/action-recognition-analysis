from dotenv import load_dotenv, find_dotenv
import os

env_path = find_dotenv()
load_dotenv(env_path)


TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_PHONE = os.getenv("TWILIO_FROM_PHONE")
RIGGUARDIAN_WEBHOOK_URL = os.getenv("RIGGUARDIAN_WEBHOOK_URL")
TO_PHONE_NUMBERS = os.getenv("TO_PHONE_NUMBERS")
RUNTIME_ENV = os.getenv("RUNTIME_ENV", "dev").strip().lower()
ENABLE_UI = os.getenv("ENABLE_UI", "1" if RUNTIME_ENV == "dev" else "0")
ENABLE_SMS = os.getenv("ENABLE_SMS", "0" if RUNTIME_ENV == "dev" else "1")
STREAM_SOURCES_FILE_PATH = os.getenv("STREAM_SOURCES_FILE_PATH")
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "yolo26n-pose.pt")
HUMAN_DETECTOR_MODEL_PATH = os.getenv("HUMAN_DETECTOR_MODEL_PATH", "yolo26m.pt")
HUMAN_DETECTOR_CANDIDATE_MODELS = [
    item.strip()
    for item in os.getenv(
        "HUMAN_DETECTOR_CANDIDATE_MODELS",
        "yolo26m_fp16.engine,yolo26m.engine,yolo26m.pt,yolo26m-pose.pt",
    ).split(",")
    if item.strip()
]
