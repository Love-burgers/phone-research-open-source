"""LangChain chat-model adapter, preserving the existing JSON client protocol."""

from urllib.parse import urlsplit
from contextvars import ContextVar

from langchain_core.output_parsers import JsonOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from .execution_context import request_timeout
from .usage_guard import provider_quota

from .config import (
    MODEL_API_URL,
    MODEL_MAX_TOKENS,
    MODEL_NAME,
    MODEL_TIMEOUT_SECONDS,
    get_model_api_key,
    get_settings,
)


def model_base_url(endpoint: str) -> str:
    """Accept both the existing full endpoint and an SDK base URL."""
    return endpoint.rstrip("/").removesuffix("/chat/completions")


class ModelClient:
    """Use a LangChain model and JSON parser; application validates citations."""

    def __init__(self, chat_model=None) -> None:
        self.chat_model = chat_model
        self._usage = ContextVar('model_usage',default=None)

    @property
    def last_usage(self):
        return self._usage.get()

    @last_usage.setter
    def last_usage(self,value):
        self._usage.set(value)

    @property
    def available(self) -> bool:
        return self.chat_model is not None or bool(get_model_api_key())

    def _model(self):
        key = get_model_api_key()
        if not key:
            raise ValueError("未配置 MODEL_API_KEY")

        settings=get_settings()
        base_url = model_base_url(settings.model_api_url)
        provider = settings.model_provider or (
            "deepseek" if urlsplit(base_url).hostname == "api.deepseek.com" else "openai-compatible"
        )
        options = dict(model=settings.model_name, api_key=key, base_url=base_url,
                       timeout=request_timeout(settings.model_timeout_seconds), max_retries=0, max_tokens=settings.model_max_tokens)
        if provider == "deepseek":
            return ChatDeepSeek(**options)
        if provider == "openai-compatible":
            return ChatOpenAI(**options, use_responses_api=False)
        raise ValueError("MODEL_PROVIDER 必须为 deepseek 或 openai-compatible")

    def _record_usage(self, response):
        usage = response.usage_metadata or {}
        self.last_usage = {
            target: value for source, target in (
                ("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
            ) if isinstance((value := usage.get(source)), int) and not isinstance(value, bool) and value >= 0
        } or None

    def invoke_tools(self, messages, tools, *, max_output_tokens: int = 1200):
        """Return the model's tool requests; execution belongs to the guarded graph."""
        self.last_usage = None
        with provider_quota([*messages, {"tools":tools}],max_output_tokens) as usage:
            model = self.chat_model if self.chat_model is not None else self._model()
            response = model.bind_tools(tools).bind(max_tokens=max_output_tokens).invoke(messages)
            self._record_usage(response)
            usage.update(self.last_usage or {})
        return response

    def complete_json(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> dict:
        self.last_usage = None
        options = {"response_format": {"type": "json_object"}}
        if max_output_tokens is not None:
            options["max_tokens"] = max_output_tokens
        with provider_quota(messages,max_output_tokens or MODEL_MAX_TOKENS) as usage:
            model = self.chat_model if self.chat_model is not None else self._model()
            response = model.bind(**options).invoke(messages)
            self._record_usage(response)
            usage.update(self.last_usage or {})
        result = JsonOutputParser().invoke(response)
        if not isinstance(result, dict):
            raise ValueError("模型必须返回 JSON 对象")
        return result

