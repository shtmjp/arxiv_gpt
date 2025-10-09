from __future__ import annotations

from pathlib import Path

from explain import ExplanationPDFRenderer, MessageStore, PostedPaper


def test_message_store_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    store = MessageStore(path)
    paper = PostedPaper(paper_id="http://arxiv.org/abs/1234.5678v1", channel_id="10", message_id="20")
    store.add_posted_paper(paper)
    store.mark_request_handled("30")

    reloaded = MessageStore(path)
    stored_papers = list(reloaded.iter_posted_papers())
    assert stored_papers == [paper]
    assert reloaded.is_request_handled("30")
    assert not reloaded.is_request_handled("999")


def test_pdf_renderer(tmp_path: Path) -> None:
    renderer = ExplanationPDFRenderer()
    pdf_bytes = renderer.render_to_bytes("サンプル\nテキスト")
    output = tmp_path / "out.pdf"
    output.write_bytes(pdf_bytes)
    assert output.exists()
    assert output.stat().st_size > 0
