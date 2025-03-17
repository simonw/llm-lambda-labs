import click
from httpx_sse import connect_sse, aconnect_sse
import httpx
import json
import llm
from pydantic import Field
from typing import Optional


def fetch_models(key=None):
    key = llm.get_key(key or "", "lambdalabs", "LLM_LAMBDALABS_KEY")
    if not key:
        raise click.ClickException(
            "You must set the 'lambdalabs' key or the LLM_LAMBDALABS_KEY environment variable."
        )
    try:
        response = httpx.get(
            "https://api.lambdalabs.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"]
    except httpx.HTTPStatusError as e:
        raise click.ClickException(f"Error fetching models: {e.response.text}")


def get_model_details(key=None, force_fetch=False):
    user_dir = llm.user_dir()
    lambdalabs_models = user_dir / "lambdalabs_models.json"
    if lambdalabs_models.exists() and not force_fetch:
        models = json.loads(lambdalabs_models.read_text())
    else:
        models = fetch_models(key=key)
        lambdalabs_models.write_text(json.dumps(models, indent=2))
    return models


@llm.hookimpl
def register_models(register):
    for model in get_model_details():
        model_id = model["id"]
        our_model_id = "lambdalabs/" + model_id
        register(
            LambdaLabs(our_model_id, model_id), AsyncLambdaLabs(our_model_id, model_id)
        )


def get_model_ids(key):
    return [model["id"] for model in get_model_details(key)]


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def lambdalabs():
        "Commands relating to the llm-lambda-labs plugin"

    @lambdalabs.command()
    @click.option("--key", help="Lambda Labs API key")
    def refresh(key):
        "Refresh the list of available Lambda Labs models"
        before = set(get_model_ids(key=key))
        get_model_details(key=key, force_fetch=True)
        after = set(get_model_ids(key=key))
        added = after - before
        removed = before - after
        if added:
            click.echo(f"Added models: {', '.join(added)}", err=True)
        if removed:
            click.echo(f"Removed models: {', '.join(removed)}", err=True)
        if added or removed:
            click.echo("New list of models:", err=True)
            for model_id in get_model_ids(key=key):
                click.echo(model_id, err=True)
        else:
            click.echo("No changes", err=True)

    @lambdalabs.command()
    @click.option("--key", help="Lambda Labs API key")
    def models(key):
        "List available Lambda Labs models"
        details = get_model_details(key)
        click.echo(json.dumps(details, indent=2))


class _SharedLambdaLabs:
    can_stream = True
    needs_key = "lambdalabs"
    key_env_var = "LLM_LAMBDALABS_KEY"

    class Options(llm.Options):
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate in the completion.",
            ge=0,
            default=None,
        )

    def __init__(self, our_model_id, lambda_labs_id):
        self.model_id = our_model_id
        self.lambda_labs_id = lambda_labs_id

    def __str__(self):
        return "Lambda Labs: {}".format(self.model_id)

    def build_messages(self, prompt, conversation):
        messages = []
        latest_message = {"role": "user", "content": prompt.prompt}
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append(latest_message)
            return messages

        system_from_conversation = None
        for prev_response in conversation.responses:
            if not prompt.system and prev_response.prompt.system:
                system_from_conversation = prev_response.prompt.system
            messages.append({"role": "user", "content": prev_response.prompt.prompt})
            messages.append({"role": "assistant", "content": prev_response.text()})

        if system_from_conversation:
            messages = [{"role": "system", "content": prompt.system}] + messages

        messages.append(latest_message)
        return messages

    def build_request_body(self, prompt, conversation):
        messages = self.build_messages(prompt, conversation)
        body = {
            "model": self.lambda_labs_id,
            "messages": messages,
        }
        if prompt.options.max_tokens:
            body["max_tokens"] = prompt.options.max_tokens
        return body

    def set_usage(self, response, usage):
        if usage:
            response.set_usage(
                input=usage["prompt_tokens"],
                output=usage["completion_tokens"],
            )


class LambdaLabs(_SharedLambdaLabs, llm.KeyModel):
    def execute(self, prompt, stream, response, conversation, key=None):
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        body = self.build_request_body(prompt, conversation)

        if stream:
            body["stream"] = True
            with httpx.Client() as client:
                with connect_sse(
                    client,
                    "POST",
                    "https://api.lambdalabs.com/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                ) as event_source:
                    # In case of unauthorized:
                    if event_source.response.status_code != 200:
                        raise ValueError(str(event_source.response.status_code))
                    event_source.response.raise_for_status()
                    last_not_done = None
                    for sse in event_source.iter_sse():
                        if sse.data != "[DONE]":
                            try:
                                data = sse.json()
                                last_not_done = data
                                yield data["choices"][0]["delta"]["content"]
                            except KeyError:
                                pass
                    # Record last_not_done as response_json - it includes usage
                    if last_not_done:
                        last_not_done.pop("choices", None)
                        self.set_usage(response, last_not_done.pop("usage", None))
                        response.response_json = last_not_done
        else:
            with httpx.Client() as client:
                api_response = client.post(
                    "https://api.lambdalabs.com/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                )
                api_response.raise_for_status()
                yield api_response.json()["choices"][0]["message"]["content"]
                details = api_response.json()
                self.set_usage(response, details.pop("usage", None))
                response.response_json = details


class AsyncLambdaLabs(_SharedLambdaLabs, llm.AsyncKeyModel):
    async def execute(self, prompt, stream, response, conversation, key=None):
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}
        body = self.build_request_body(prompt, conversation)

        if stream:
            body["stream"] = True
            async with httpx.AsyncClient() as client:
                async with aconnect_sse(
                    client,
                    "POST",
                    "https://api.lambdalabs.com/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                ) as event_source:
                    # In case of unauthorized:
                    if event_source.response.status_code != 200:
                        raise ValueError(str(event_source.response.status_code))
                    event_source.response.raise_for_status()
                    last_not_done = None
                    async for sse in event_source.aiter_sse():
                        if sse.data != "[DONE]":
                            try:
                                data = sse.json()
                                last_not_done = data
                                yield data["choices"][0]["delta"]["content"]
                            except KeyError:
                                pass
                    # Record last_not_done as response_json - it includes usage
                    if last_not_done:
                        last_not_done.pop("choices", None)
                        self.set_usage(response, last_not_done.pop("usage", None))
                        response.response_json = last_not_done
        else:
            async with httpx.AsyncClient() as client:
                api_response = await client.post(
                    "https://api.lambdalabs.com/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                    json=body,
                    timeout=None,
                )
                api_response.raise_for_status()
                response_json = api_response.json()
                yield response_json["choices"][0]["message"]["content"]
                self.set_usage(response, response_json.pop("usage", None))
                response.response_json = response_json
