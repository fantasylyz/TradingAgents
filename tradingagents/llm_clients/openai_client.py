import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI
import httpx
from httpx._client import _DEFAULT_TIMEOUT_CONFIG

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        """Wrap with structured output, defaulting to function_calling for OpenAI.

        langchain-openai's Responses-API-parse path (the default for json_schema
        when use_responses_api=True) calls response.model_dump(...) on the OpenAI
        SDK's union-typed parsed response, which makes Pydantic emit ~20
        PydanticSerializationUnexpectedValue warnings per call. The function-calling
        path returns a plain tool-call shape that does not trigger that
        serialization, so it is the cleaner choice for our combination of
        use_responses_api=True + with_structured_output. Both paths use OpenAI's
        strict mode and produce the same typed Pydantic instance.
        """
        if method is None:
            method = "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm": ("https://api.z.ai/api/paas/v4/", "ZHIPU_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}

# Provider-specific defaults for reliability
_PROVIDER_DEFAULTS = {
    "deepseek": {
        "max_retries": 3,
        "timeout": 120.0,  # DeepSeek can be slow; give it 2 minutes
    },
    "qwen": {
        "max_retries": 3,
        "timeout": 120.0,
    },
}


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        # Provider-specific base URL and auth
        if self.provider in _PROVIDER_CONFIG:
            base_url, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = base_url
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
                else:
                    raise ValueError(
                        f"API key for provider '{self.provider}' is required but not set. "
                        f"Please set the {api_key_env} environment variable."
                    )
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Apply provider-specific defaults if not already set by user
        if self.provider in _PROVIDER_DEFAULTS:
            defaults = _PROVIDER_DEFAULTS[self.provider]
            for key, default_value in defaults.items():
                # User config takes precedence over defaults
                if key not in self.kwargs and key not in llm_kwargs:
                    llm_kwargs[key] = default_value

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Create custom http_client with proper retry configuration for problematic providers
        # This is critical for DeepSeek which has unstable connections
        if self.provider in _PROVIDER_DEFAULTS and "http_client" not in llm_kwargs:
            timeout_val = llm_kwargs.get("timeout", 120.0)
            max_retries_val = llm_kwargs.get("max_retries", 3)
            
            # Convert timeout to httpx.Timeout format
            if isinstance(timeout_val, (int, float)):
                timeout = httpx.Timeout(timeout_val, connect=timeout_val, read=timeout_val, write=timeout_val, pool=timeout_val)
            else:
                timeout = timeout_val
            
            # Create retry object for httpx
            retry_transport = httpx.HTTPTransport(
                retries=max_retries_val,
                timeout=timeout,
            )
            
            # Create http_client with retry configuration
            http_client = httpx.Client(
                transport=retry_transport,
                timeout=timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
            llm_kwargs["http_client"] = http_client

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
