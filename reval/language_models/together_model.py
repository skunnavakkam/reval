from together import Together, AsyncTogether
import os


class TogetherChatModel:
    def __init__(
        self, model_name, system_prompt=None, max_tokens=4096, temperature=0.5
    ):
        self.model_name = model_name
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(self, prompt):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        return (
            self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            .choices[0]
            .message.content
        )


class TogetherBaseModel:
    def __init__(self, model_name, max_tokens=4096, temperature=0.5):
        self.model_name = model_name
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(self, prompt):
        return (
            self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            .choices[0]
            .text
        )


class AsyncTogetherChatModel:
    def __init__(
        self, model_name, system_prompt=None, max_tokens=4096, temperature=0.5
    ):
        self.model_name = model_name
        self.client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

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


class AsyncTogetherBaseModel:
    def __init__(self, model_name, max_tokens=4096, temperature=0.5):
        self.model_name = model_name
        self.client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def __call__(self, prompt):
        response = await self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].text
