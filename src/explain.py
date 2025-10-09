"""Utilities for responding to detailed explanation requests on Discord."""

from __future__ import annotations

import io
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import arxiv
import google.generativeai as genai
import requests
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


@dataclass(frozen=True, slots=True)
class PostedPaper:
    """Metadata about a paper summary that has been posted to Discord."""

    paper_id: str
    channel_id: str
    message_id: str


class MessageStore:
    """Persist information about posted papers and handled requests."""

    _POSTED_KEY = "posted_papers"
    _HANDLED_KEY = "handled_request_ids"

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data = {
            self._POSTED_KEY: [],
            self._HANDLED_KEY: [],
        }
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self._data.update({
                    self._POSTED_KEY: loaded.get(self._POSTED_KEY, []),
                    self._HANDLED_KEY: loaded.get(self._HANDLED_KEY, []),
                })

    def _dump(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def add_posted_paper(self, paper: PostedPaper) -> None:
        """Register a newly posted paper summary."""

        posted: list[dict[str, str]] = self._data[self._POSTED_KEY]
        if any(entry.get("message_id") == paper.message_id for entry in posted):
            return
        posted.append(
            {
                "paper_id": paper.paper_id,
                "channel_id": paper.channel_id,
                "message_id": paper.message_id,
            }
        )
        self._dump()

    def iter_posted_papers(self) -> Iterator[PostedPaper]:
        """Iterate over registered posted papers."""

        for entry in self._data[self._POSTED_KEY]:
            try:
                yield PostedPaper(
                    paper_id=str(entry["paper_id"]),
                    channel_id=str(entry["channel_id"]),
                    message_id=str(entry["message_id"]),
                )
            except KeyError:
                continue

    def mark_request_handled(self, request_id: str) -> None:
        handled: list[str] = self._data[self._HANDLED_KEY]
        if request_id in handled:
            return
        handled.append(request_id)
        self._dump()

    def is_request_handled(self, request_id: str) -> bool:
        handled: list[str] = self._data[self._HANDLED_KEY]
        return request_id in handled


def _shorten_entry_id(entry_id: str) -> str:
    """Convert an arXiv entry id URL into its short identifier."""

    return entry_id.rsplit("/", maxsplit=1)[-1]


class ExplanationGenerator:
    """Generate structured explanations for a paper using Gemini."""

    _PROMPT = (
        "以下のPDFに含まれる論文を読み、次の4点を日本語で詳しく説明してください。\n"
        "1. 既存研究と比較した新規性\n"
        "2. モデルで使われている数式を交えた説明\n"
        "3. モデルのインプットとアウトプットの形式\n"
        "4. 実験に用いられたデータ形式の詳細\n"
        "各項目は見出しと箇条書きを含むMarkdown形式で出力してください。"
    )

    def __init__(self, model_name: str = "gemini-2.5-pro") -> None:
        self._model = genai.GenerativeModel(model_name)
        self._client = arxiv.Client()

    def _download_pdf(self, entry_id: str) -> Path:
        search = arxiv.Search(id_list=[entry_id])
        results = list(self._client.results(search))
        if not results:
            msg = f"Paper with id '{entry_id}' was not found on arXiv."
            raise ValueError(msg)

        result = results[0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "paper.pdf"
            result.download_pdf(dirpath=tmp_dir, filename="paper.pdf")
            # Copy to a stable temporary file outside the context manager.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                with pdf_path.open("rb") as source:
                    tmp_pdf.write(source.read())
                return Path(tmp_pdf.name)

    def generate(self, paper_id: str) -> str:
        entry_id = _shorten_entry_id(paper_id)
        temp_pdf = self._download_pdf(entry_id)
        uploaded_file = None
        try:
            uploaded_file = genai.upload_file(path=str(temp_pdf))
            response = self._model.generate_content(
                [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "file_data": {
                                    "file_uri": uploaded_file.uri,
                                    "mime_type": "application/pdf",
                                }
                            },
                            {"text": self._PROMPT},
                        ],
                    }
                ]
            )
            if not getattr(response, "text", None):  # pragma: no cover - runtime safeguard
                raise RuntimeError("Gemini did not return any text for the explanation.")
            return response.text.strip()
        finally:
            try:
                if uploaded_file is not None:
                    genai.delete_file(uploaded_file.name)
            finally:
                try:
                    os.unlink(temp_pdf)
                except FileNotFoundError:
                    pass


class ExplanationPDFRenderer:
    """Render explanation text into a simple PDF document."""

    def __init__(self, font_name: str = "HeiseiKakuGo-W5", font_size: int = 12) -> None:
        self._font_name = font_name
        self._font_size = font_size
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))

    def _wrap_text(self, text: str, max_chars: int = 40) -> Iterable[str]:
        for paragraph in text.splitlines():
            if not paragraph:
                yield ""
                continue
            line = ""
            for char in paragraph:
                line += char
                if len(line) >= max_chars:
                    yield line
                    line = ""
            if line:
                yield line

    def render_to_bytes(self, text: str) -> bytes:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        pdf.setFont(self._font_name, self._font_size)

        width, height = A4
        margin = 40
        leading = self._font_size * 1.5
        y = height - margin

        for line in self._wrap_text(text):
            if y <= margin:
                pdf.showPage()
                pdf.setFont(self._font_name, self._font_size)
                y = height - margin
            if line:
                pdf.drawString(margin, y, line)
            y -= leading

        pdf.save()
        buffer.seek(0)
        return buffer.read()


class DiscordAPI:
    """Minimal Discord REST client for fetching replies and posting files."""

    def __init__(self, bot_token: str) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bot {bot_token}",
                "User-Agent": "arxiv-gpt-bot",
            }
        )

    def fetch_replies(self, channel_id: str, message_id: str) -> list[dict[str, object]]:
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        response = self._session.get(url, params={"limit": 100})
        response.raise_for_status()
        messages: list[dict[str, object]] = response.json()
        replies = []
        for message in messages:
            ref = message.get("message_reference")
            if isinstance(ref, dict) and ref.get("message_id") == message_id:
                replies.append(message)
        return replies

    def post_pdf_reply(
        self,
        channel_id: str,
        reply_to: str,
        filename: str,
        file_bytes: bytes,
        content: str,
    ) -> None:
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        payload = {
            "content": content,
            "message_reference": {"message_id": reply_to},
        }
        files = {
            "files[0]": (filename, file_bytes, "application/pdf"),
        }
        data = {"payload_json": json.dumps(payload)}
        response = self._session.post(url, data=data, files=files)
        response.raise_for_status()


def process_explain_requests(
    store: MessageStore,
    discord_api: DiscordAPI,
    generator: ExplanationGenerator,
    renderer: ExplanationPDFRenderer,
) -> None:
    """Process all outstanding explanation requests."""

    for posted in store.iter_posted_papers():
        replies = discord_api.fetch_replies(posted.channel_id, posted.message_id)
        for reply in replies:
            request_id = str(reply.get("id"))
            if store.is_request_handled(request_id):
                continue
            content = str(reply.get("content", ""))
            if "解説して" not in content:
                continue
            explanation = generator.generate(posted.paper_id)
            pdf_bytes = renderer.render_to_bytes(explanation)
            filename = f"{_shorten_entry_id(posted.paper_id)}-explanation.pdf"
            discord_api.post_pdf_reply(
                channel_id=posted.channel_id,
                reply_to=request_id,
                filename=filename,
                file_bytes=pdf_bytes,
                content="論文の解説をPDFでお届けします。",
            )
            store.mark_request_handled(request_id)
