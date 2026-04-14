from dotenv import load_dotenv, find_dotenv
import os

env_path = find_dotenv()
load_dotenv(env_path)


TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_PHONE = os.getenv("TWILIO_FROM_PHONE")
RIGGUARDIAN_WEBHOOK_URL = os.getenv("RIGGUARDIAN_WEBHOOK_URL")
TO_PHONE_NUMBERS = os.getenv("TO_PHONE_NUMBERS")
ENABLE_UI = os.getenv("ENABLE_UI")
STREAM_SOURCES_FILE_PATH = os.getenv("STREAM_SOURCES_FILE_PATH")
