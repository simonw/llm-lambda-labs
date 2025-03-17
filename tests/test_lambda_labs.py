import llm


def test_plugin_is_installed():
    models = llm.get_models_with_aliases()
    model_ids = [model.model.model_id for model in models]
    assert any(model_id.startswith("lambdalabs/") for model_id in model_ids)
