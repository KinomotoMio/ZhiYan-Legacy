from __future__ import annotations

from pathlib import Path

import pytest

from app.services import slidev as slidev_mod


def _real_failure_fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "slidev" / "composition_repair_slides.md"


def test_real_slidev_failure_fixture_normalizes_to_five_pages() -> None:
    fixture_path = _real_failure_fixture_path()
    assert fixture_path.exists(), f"missing fixture: {fixture_path}"
    markdown = fixture_path.read_text(encoding="utf-8")

    inspection = slidev_mod.inspect_slidev_markdown_submission(markdown=markdown)
    parsed = slidev_mod.parse_slidev_markdown(markdown=markdown)

    assert inspection["raw_slide_count"] == 9
    assert inspection["normalized_slide_count"] == 5
    assert parsed["slide_count"] == 5
    assert [slide["title"] for slide in parsed["slides"]] == [
        "Prompt，不只是文字",
        "藏在提示词背后的三次转向",
        "PART 01",
        "协作者模式：沉默是金",
        "四岁小孩就会的事",
    ]


@pytest.mark.asyncio
async def test_prepare_slidev_deck_artifact_preserves_markdown_and_soft_page_count(monkeypatch) -> None:
    markdown = (
        "---\n"
        "title: Slidev Soft Count\n"
        "---\n\n"
        "# 封面\n\n"
        "---\n\n"
        "# 第 2 页\n"
    )

    async def _unexpected_validate(**kwargs):  # noqa: ANN003
        raise AssertionError("validate_slidev_deck should not be called")

    async def _unexpected_review(**kwargs):  # noqa: ANN003
        raise AssertionError("review_slidev_deck should not be called")

    monkeypatch.setattr(slidev_mod, "validate_slidev_deck", _unexpected_validate)
    monkeypatch.setattr(slidev_mod, "review_slidev_deck", _unexpected_review)

    prepared = await slidev_mod.prepare_slidev_deck_artifact(
        markdown=markdown,
        fallback_title="Slidev Soft Count",
        selected_style_id="tech-launch",
        topic="Slidev Soft Count",
        outline_items=[],
        expected_pages=3,
    )

    assert prepared["markdown"] == markdown
    assert prepared["selected_style_id"] == "tech-launch"
    assert prepared["meta"]["slide_count"] == 2
    assert prepared["meta"]["selected_style_id"] == "tech-launch"
    assert prepared["meta"]["page_count_check"] == {
        "expected_slide_count": 3,
        "submitted_slide_count": 2,
        "matches_expected": False,
        "mode": "soft",
    }
