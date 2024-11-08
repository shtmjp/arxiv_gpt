from pathlib import Path

import gspread

from log import get_logger_for_test
from settings import SHEET_ID


def test_gs() -> None:
    gc = gspread.service_account(  # type: ignore[attr-defined]
        filename=Path(__file__).resolve().parents[1] / "credentials/gs_credentials.json"
    )
    sh = gc.open_by_key(SHEET_ID)
    worksheet = sh.get_worksheet(0)
    val = worksheet.acell("A1").value
    logger = get_logger_for_test(__name__)
    logger.debug(val)
