import os
from pathlib import Path

from dotenv import load_dotenv

# renderの本番環境では.envは存在しないため、存在する場合のみ読み込む
# 本番環境では手動で環境変数を設定しておく
_dotenv_path = Path(__file__).resolve().parent / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
DEBUG_DISCORD_WEBHOOK_URL = os.environ["DEBUG_DISCORD_WEBHOOK_URL"]
SHEET_ID = os.environ["SHEET_ID"]
