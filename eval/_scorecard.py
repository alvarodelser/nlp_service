"""
Shared config for eval notebooks.
Set NLP_SERVICE_URL env var to point at a non-local instance.
"""
import os

NLP_BASE_URL = os.getenv("NLP_SERVICE_URL", "http://localhost:8000")
HEADERS = {"Content-Type": "application/json"}
