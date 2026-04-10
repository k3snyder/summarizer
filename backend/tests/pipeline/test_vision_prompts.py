from pathlib import Path


def test_vision_classifier_prompt_targets_substantive_visuals():
    prompt_path = (
        Path(__file__).resolve().parents[2] / "prompts" / "vision-classifier.txt"
    )
    prompt = prompt_path.read_text(encoding="utf-8")

    assert "substantive visual content worth running detailed visual extraction on" in prompt
    assert "materially lost if we relied on normal text extraction alone" in prompt
    assert "Header or footer logos" in prompt
    assert "decorative branding/background art should be NO" in prompt
    assert "A text-heavy page with only a footer logo should be NO." in prompt
    assert "Return exactly one word in uppercase" in prompt
