import os
from pathlib import Path

import deepeval
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

deepeval.login(os.environ["CONFIDENT_API_KEY"])

