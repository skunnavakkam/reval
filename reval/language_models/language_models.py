from anthropic import Anthropic
from openai import OpenAI
from together import Together
import os


class LanguageModel:
    def __init__(self, model_name, name=None, system_prompt=None):
        self.system_prompt = system_prompt
        claude_disambiguations = {
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-sonnet": "claude-3-sonnet-20240307",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
        }
        if name is None:
            self.name = model_name
        else:
            self.name = name

        # get model_provider
        if "/" in model_name:
            model_provider = "together"
            self.model = model_name
            self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        elif "claude" in model_name:
            model_provider = "claude"
            if model_name in claude_disambiguations:
                self.model = claude_disambiguations[model_name]
            else:
                self.model = model_name
            self.client = Anthropic()
        elif "gpt" in model_name:
            model_provider = "openai"
            self.model = model_name
            self.client = OpenAI()
        else:
            raise (NameError(f"Model {model_name} not recognized"))

    def get_generation(self, prompt) -> str:
        if self.system_prompt is not None:
            if self.model_provider == "claude":
                return (
                    self.client.messages.create(
                        model=self.model,
                        system_prompt=self.system_prompt,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=4096,
                        temperature=1,
                    )
                    .content[0]
                    .text
                )
            elif self.model_provider == "openai":
                return (
                    self.client.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=4096,
                        temperature=1,
                    )
                    .choices[0]
                    .message.content
                )
            elif self.model_provider == "together":
                # uses same schema as open
                return (
                    self.client.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=4096,
                        temperature=1,
                    )
                    .choices[0]
                    .message.content
                )

        else:
            if self.model_provider == "claude":
                return (
                    self.client.messages.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=4096,
                        temperature=1,
                    )
                    .content[0]
                    .text
                )
            elif self.model_provider == "openai":
                return (
                    self.client.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=4096,
                        temperature=1,
                    )
                    .choices[0]
                    .message.content
                )
            elif self.model_provider == "together":
                # uses same schema as open
                return (
                    self.client.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=4096,
                        temperature=1,
                    )
                    .choices[0]
                    .message.content
                )
