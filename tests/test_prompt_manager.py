import pytest

from services.prompt_manager import PromptManager


def test_get_caches_prompt_until_reload(tmp_path):
    prompt_file = tmp_path / "main.txt"
    prompt_file.write_text("  first prompt  ", encoding="utf-8")
    manager = PromptManager(tmp_path)

    assert manager.get("main") == "first prompt"

    prompt_file.write_text("second prompt", encoding="utf-8")
    assert manager.get("main") == "first prompt"
    assert manager.reload("main") == "second prompt"


def test_get_raises_for_missing_prompt(tmp_path):
    manager = PromptManager(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing.txt"):
        manager.get("missing")

