from anthropic import Anthropic, AsyncAnthropic


class AnthropicModel:
    def __init__(
        self, model_name, system_prompt=None, max_tokens=4096, temperature=0.5
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = Anthropic()

    def __call__(self, prompt):
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.system_prompt is not None:
            kwargs["system"] = self.system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class AsyncAnthropicModel:
    def __init__(
        self, model_name, system_prompt=None, max_tokens=4096, temperature=0.5
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AsyncAnthropic()

    async def __call__(self, prompt):
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.system_prompt is not None:
            kwargs["system"] = self.system_prompt

        response = await self.client.messages.create(**kwargs)
        return response.content[0].text
