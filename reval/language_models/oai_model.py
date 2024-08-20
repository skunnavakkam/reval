from openai import OpenAI, AsyncOpenAI


class OpenAIModel:
    def __init__(
        self, model_name, system_prompt=None, max_tokens=4096, temperature=0.5
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI()

    def __call__(self, prompt):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response["choices"][0]["message"]["content"]


class AsyncOpenAIModel:
    def __init__(
        self, model_name, system_prompt=None, max_tokens=4096, temperature=0.5
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AsyncOpenAI()

    async def __call__(self, prompt):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
