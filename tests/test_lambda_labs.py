import json
import llm
import os
import pytest

API_KEY = os.environ.get("PYTEST_LAMBDALABS_KEY", None) or "key-..."


@pytest.fixture
def patched(monkeypatch, tmpdir):
    monkeypatch.setenv("LLM_LAMBDALABS_KEY", API_KEY)
    monkeypatch.setenv("LLM_USER_PATH", str(tmpdir))
    (tmpdir / "lambdalabs_models.json").write_text(
        json.dumps(
            [
                {
                    "id": "llama3.3-70b-instruct-fp8",
                    "object": "model",
                    "created": 1724347380,
                    "owned_by": "lambda",
                }
            ]
        ),
        "utf-8",
    )


def test_plugin_is_installed(patched):
    models = llm.get_models_with_aliases()
    model_ids = [model.model.model_id for model in models]
    assert any(model_id.startswith("lambdalabs/") for model_id in model_ids)


@pytest.mark.vcr
def test_prompt(patched):
    model = llm.get_model("lambdalabs/llama3.3-70b-instruct-fp8")
    output = model.prompt("hi", stream=False).text()
    assert (
        output
        == "It's nice to meet you. Is there something I can help you with or would you like to chat?"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (True, False))
async def test_async_prompt(patched, stream):
    model = llm.get_async_model("lambdalabs/llama3.3-70b-instruct-fp8")
    output = await model.prompt("hi", stream=stream).text()
    assert (
        output
        == "How's it going? Is there something I can help you with or would you like to chat?"
    )
