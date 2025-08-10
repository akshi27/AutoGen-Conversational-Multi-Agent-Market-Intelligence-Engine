import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys from .env
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

# Other Configurations
BRANDS = [
    "atomberg", "usha", "havells", "crompton", "orient", "bajaj", "luminous", "lg", "vguard", "honeywell", "polycab"
]
PRODUCT_TYPES = [
    "smart fan", "bldc fan", "ceiling fan", "fan"
]
SUBREDDITS = ["smarthome", "homeautomation", "india", "mumbai", "Chennai", "Frugal_Ind", "Kerala", "delhi", "kolkata"]
TARGET_LANG = "en"