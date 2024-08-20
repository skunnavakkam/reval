from reval.language_models import OpenAIModel, AnthropicModel, TogetherChatModel
from reval.language_models import (
    AsyncAnthropicModel,
    AsyncOpenAIModel,
    AsyncTogetherChatModel,
)


def GenericLanguageModel(
    model_name, system_prompt=None, max_tokens=4096, temperature=0.5
):
    if "gpt" in model_name:
        return OpenAIModel(model_name, system_prompt, max_tokens, temperature)
    elif "claude" in model_name:
        return AnthropicModel(model_name, system_prompt, max_tokens, temperature)
    else:
        return TogetherChatModel(model_name, system_prompt, max_tokens, temperature)


def AsyncGenericLanguageModel(
    model_name, system_prompt=None, max_tokens=4096, temperature=0.5
):
    if "gpt" in model_name:
        return AsyncOpenAIModel(model_name, system_prompt, max_tokens, temperature)
    elif "claude" in model_name:
        return AsyncAnthropicModel(model_name, system_prompt, max_tokens, temperature)
    else:
        return AsyncTogetherChatModel(
            model_name, system_prompt, max_tokens, temperature
        )
