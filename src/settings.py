import os
from pathlib import Path

from dotenv import load_dotenv

# renderの本番環境では.envは存在しないため、存在する場合のみ読み込む
# 本番環境では手動で環境変数を設定しておく
_dotenv_path = Path(__file__).resolve().parent / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
# localではDISCORD_WEBHOOK_URLはデバッグ用URLを設定しておく
# 本番環境では別に設定
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
SHEET_ID = os.environ["SHEET_ID"]

# 複数クエリの設定をJSON文字列で受け取るための環境変数
# 例:
#   export EXTRA_ARXIV_SEARCH_CONFIGS='[
#       {"query": "cat:stat.ML", "webhook_env": "DISCORD_WEBHOOK_URL_STAT"},
#       {"query": "ti:\"lead lag\"", "webhook_url": "https://discord..."}
#   ]'
EXTRA_ARXIV_SEARCH_CONFIGS = os.environ.get("EXTRA_ARXIV_SEARCH_CONFIGS")
