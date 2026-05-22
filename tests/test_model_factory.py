from services import model_factory


def test_get_model_name_uses_openai_settings(monkeypatch):
    monkeypatch.setattr(model_factory.settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(model_factory.settings, "OPENAI_MODEL_MAIN", "openai-main")
    monkeypatch.setattr(model_factory.settings, "OPENAI_MODEL_VISION", "openai-vision")

    assert model_factory.get_model_name() == "openai-main"
    assert model_factory.get_model_name("vision") == "openai-vision"


def test_get_model_name_uses_gemini_settings(monkeypatch):
    monkeypatch.setattr(model_factory.settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(model_factory.settings, "GEMINI_MODEL_MAIN", "gemini-main")
    monkeypatch.setattr(model_factory.settings, "GEMINI_MODEL_VISION", "gemini-vision")

    assert model_factory.get_model_name() == "gemini-main"
    assert model_factory.get_model_name("vision") == "gemini-vision"



