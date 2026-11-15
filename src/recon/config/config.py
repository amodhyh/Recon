import os
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, PrivateAttr

# Load environment variables
_project_root = Path(__file__).resolve().parents[3]
_env_path = _project_root / ".env"
if _env_path.is_file():
    load_dotenv(_env_path, override=False)
else:
    load_dotenv(find_dotenv(), override=False)


class BaseLLMModel(BaseModel):
    """Base class wrapping a LangChain BaseChatModel for Recon agents using Pydantic."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    model_name: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None

    _model: BaseChatModel | None = PrivateAttr(default=None)

    def _build_model(self) -> BaseChatModel:
        """Abstract builder to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_model")

    def get_model(self) -> BaseChatModel:
        """Return the initialized BaseChatModel instance."""
        if self._model is None:
            self._model = self._build_model()
        return self._model

    @property
    def model(self) -> BaseChatModel:
        """Property shortcut to get the underlying chat model."""
        return self.get_model()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute and method calls to the underlying model."""
        if name.startswith("_") or name.startswith("__"):
            return super().__getattr__(name)
        try:
            extra = self.__pydantic_extra__
            if extra and name in extra:
                return extra[name]
        except AttributeError:
            pass
        return getattr(self.get_model(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow invoking the model instance directly."""
        return self.get_model()(*args, **kwargs)


class LlamaCPPModel(BaseLLMModel):
    """Chat model wrapper for local llama.cpp / llama-server instances."""

    host: str | None = None
    port: str | int | None = None
    base_url: str | None = None
    model_path: str | None = None
    api_key: str | None = None

    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
        port: int | str | None = None,
        base_url: str | None = None,
        model_path: str | Path | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ):
        data: dict[str, Any] = {
            "temperature": temperature,
            **kwargs,
        }
        if model_name is not None:
            data["model_name"] = model_name
        if host is not None:
            data["host"] = host
        if port is not None:
            data["port"] = port
        if base_url is not None:
            data["base_url"] = base_url
        if model_path is not None:
            data["model_path"] = str(model_path)
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if api_key is not None:
            data["api_key"] = api_key

        super().__init__(**data)

    def _build_model(self) -> BaseChatModel:
        host = self.host or os.getenv("LLAMACPP_HOST") or os.getenv("HOST") or "127.0.0.1"
        port = str(self.port or os.getenv("LLAMACPP_PORT") or os.getenv("PORT") or "8080")
        model_path = str(self.model_path or os.getenv("MODEL_PATH") or "")

        model_name = self.model_name
        if not model_name:
            if os.getenv("LLAMACPP_MODEL_NAME"):
                model_name = os.getenv("LLAMACPP_MODEL_NAME")
            elif os.getenv("LLM_NAME"):
                model_name = os.getenv("LLM_NAME")
            elif model_path:
                model_name = Path(model_path).stem
            else:
                model_name = "llama-cpp"

        if self.base_url:
            cleaned_base = self.base_url.rstrip("/")
            base_url = cleaned_base if cleaned_base.endswith("/v1") else f"{cleaned_base}/v1"
        else:
            env_url = os.getenv("LLAMACPP_BASE_URL")
            if env_url:
                cleaned_base = env_url.rstrip("/")
                base_url = cleaned_base if cleaned_base.endswith("/v1") else f"{cleaned_base}/v1"
            else:
                base_url = f"http://{host}:{port}/v1"

        api_key = self.api_key or os.getenv("LLAMACPP_API_KEY") or "no-key-required"

        extra = self.__pydantic_extra__ or {}
        init_kwargs: dict[str, Any] = {
            "model": model_name,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": self.temperature,
            **extra,
        }
        if self.max_tokens is not None:
            init_kwargs["max_tokens"] = self.max_tokens

        return ChatOpenAI(**init_kwargs)


class OpenAIModel(BaseLLMModel):
    """Chat model wrapper for OpenAI models."""

    api_key: str | None = None
    base_url: str | None = None

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ):
        data: dict[str, Any] = {
            "temperature": temperature,
            **kwargs,
        }
        if model_name is not None:
            data["model_name"] = model_name
        if api_key is not None:
            data["api_key"] = api_key
        if base_url is not None:
            data["base_url"] = base_url
        if max_tokens is not None:
            data["max_tokens"] = max_tokens

        super().__init__(**data)

    def _build_model(self) -> BaseChatModel:
        model_name = (
            self.model_name
            or os.getenv("OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL_NAME")
            or os.getenv("LLM_NAME")
            or "gpt-4o-mini"
        )
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.base_url or os.getenv("OPENAI_BASE_URL")

        extra = self.__pydantic_extra__ or {}
        init_kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": self.temperature,
            **extra,
        }
        if api_key:
            init_kwargs["api_key"] = api_key
        if base_url:
            init_kwargs["base_url"] = base_url
        if self.max_tokens is not None:
            init_kwargs["max_tokens"] = self.max_tokens

        return ChatOpenAI(**init_kwargs)


class AnthropicModel(BaseLLMModel):
    """Chat model wrapper for Anthropic Claude models."""

    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int | None = 4096

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = 4096,
        **kwargs: Any,
    ):
        data: dict[str, Any] = {
            "temperature": temperature,
            **kwargs,
        }
        if model_name is not None:
            data["model_name"] = model_name
        if api_key is not None:
            data["api_key"] = api_key
        if base_url is not None:
            data["base_url"] = base_url
        if max_tokens is not None:
            data["max_tokens"] = max_tokens

        super().__init__(**data)

    def _build_model(self) -> BaseChatModel:
        model_name = (
            self.model_name
            or os.getenv("ANTHROPIC_MODEL")
            or os.getenv("ANTHROPIC_MODEL_NAME")
            or os.getenv("LLM_NAME")
            or "claude-3-5-sonnet-latest"
        )
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        base_url = self.base_url or os.getenv("ANTHROPIC_BASE_URL")

        extra = self.__pydantic_extra__ or {}
        init_kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens or 4096,
            **extra,
        }
        if api_key:
            init_kwargs["api_key"] = api_key
        if base_url:
            init_kwargs["anthropic_api_url"] = base_url

        return ChatAnthropic(**init_kwargs)


# Aliases
LlamaCPP = LlamaCPPModel
OpenAI = OpenAIModel
Anthropic = AnthropicModel


def get_llm(
    provider: str | None = None,
    model_name: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Factory function to get a configured LLM chat model for agents.

    Args:
        provider: Provider name ('llamacpp', 'openai', 'anthropic').
                  Defaults to LLM_PROVIDER from environment, or 'llamacpp'.
        model_name: Specific model name to use.
        **kwargs: Additional parameters passed to the model constructor.

    Returns:
        BaseChatModel: Initialized LangChain chat model ready for agent use.
    """
    selected_provider = (
        provider or os.getenv("LLM_PROVIDER") or "llamacpp"
    ).lower().strip()

    if selected_provider in ("llamacpp", "llama_cpp", "llama-cpp", "llama"):
        return LlamaCPPModel(model_name=model_name, **kwargs).get_model()
    elif selected_provider in ("openai", "chatgpt", "gpt"):
        return OpenAIModel(model_name=model_name, **kwargs).get_model()
    elif selected_provider in ("anthropic", "claude"):
        return AnthropicModel(model_name=model_name, **kwargs).get_model()
    else:
        raise ValueError(
            f"Unsupported LLM provider '{selected_provider}'. "
            "Supported providers: 'llamacpp', 'openai', 'anthropic'."
        )


get_model = get_llm

__all__ = [
    "BaseLLMModel",
    "LlamaCPPModel",
    "LlamaCPP",
    "OpenAIModel",
    "OpenAI",
    "AnthropicModel",
    "Anthropic",
    "get_llm",
    "get_model",
]
