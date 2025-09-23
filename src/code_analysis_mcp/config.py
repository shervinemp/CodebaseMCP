import os

# --- LLM Configuration ---
LLM_ENABLED = os.getenv("GENERATE_LLM_DESCRIPTIONS", "false").lower() == "true"
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "5"))
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", "models/gemini-1.5-flash")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/embedding-001")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- File Watcher Configuration ---
WATCHER_POLLING_INTERVAL = int(os.getenv("WATCHER_POLLING_INTERVAL", "5"))
RESCAN_DEBOUNCE_DELAY = float(os.getenv("RESCAN_DEBOUNCE_DELAY", "5.0"))
WATCHER_PATTERNS = ["*.py"]

# --- Weaviate Configuration ---
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
