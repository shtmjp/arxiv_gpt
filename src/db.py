# 必要な機能:
# - DBに登録されている論文id一覧を取得
# - DBに論文を登録

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import arxiv
import gspread
import pandas as pd

from settings import SHEET_ID

# Implemented media (paper sources)
# Other sources like semantic scholar can be added in the future
Media = Literal["arxiv"]


@dataclass
class PaperInfo:
    """Paper information class to aggregate different paper sources."""

    medium: Media
    id: str
    authors: list[str]
    title: str
    published: datetime
    summary: str


# converter function
def arxivresult2paperinfo(result: arxiv.Result, summary: str) -> PaperInfo:
    return PaperInfo(
        medium="arxiv",
        id=result.entry_id,
        title=result.title,
        authors=[str(a) for a in result.authors],
        summary=summary,
        published=result.published,
    )


# MySQL等への移行も考慮して、抽象DAOを作成
class AbstractPaperDAO(ABC):
    """Abstract class for paper DAO.

    The database has a table with the following columns:
    - medium: str
    - id: str
    - title: str
    - authors: str
    - published: datetime
    - summary: str
    """

    @abstractmethod
    def get_paper_ids(self, medium: Media) -> list[str]:
        """Get paper ids."""

    @abstractmethod
    def add_paper(self, paper_info: PaperInfo) -> None:
        """Add paper to DB."""


class GSSPaperDAO(AbstractPaperDAO):
    """Google Spread Sheet Paper DAO."""

    def __init__(self) -> None:
        """Initialize GSSPaperDAO."""
        gc = gspread.service_account(  # type: ignore[attr-defined]
            filename=Path(__file__).resolve().parents[1] / "gs_credentials.json"
        )
        sh = gc.open_by_key(SHEET_ID)
        self.worksheet = sh.get_worksheet(0)  # worksheet for papers

    def add_paper(self, paper_info: PaperInfo) -> None:
        """Add paper to DB."""
        self.worksheet.append_row(
            [
                paper_info.medium,
                paper_info.id,
                paper_info.title,
                str(paper_info.authors),
                paper_info.published.strftime("%Y-%m-%d %H:%M:%S"),
                paper_info.summary,
            ]
        )

    def get_paper_ids(self, medium: Media) -> list[str]:
        """Get paper ids."""
        df = pd.DataFrame(self.worksheet.get_all_records())
        df = df[df["medium"] == medium]
        return df["id"].tolist()
