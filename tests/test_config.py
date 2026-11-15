import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from recon.config import (
    Anthropic,
    AnthropicModel,
    BaseLLMModel,
    LlamaCPP,
    LlamaCPPModel,
    OpenAI,
    OpenAIModel,
    get_llm,
    get_model,
)


def test_llamacpp_model_init_defaults():
    llm_wrapper = LlamaCPPModel()
    assert isinstance(llm_wrapper, BaseLLMModel)
    assert isinstance(llm_wrapper, BaseModel)
    model = llm_wrapper.get_model()
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base.endswith("/v1")
    assert llm_wrapper.model is model


def test_llamacpp_model_custom_params():
    llm_wrapper = LlamaCPPModel(
        model_name="custom-qwen",
        host="192.168.1.10",
        port=9090,
        temperature=0.2,
        max_tokens=2048,
    )
    assert isinstance(llm_wrapper, BaseModel)
    assert llm_wrapper.model_name == "custom-qwen"
    assert llm_wrapper.host == "192.168.1.10"
    assert llm_wrapper.port == 9090
    assert llm_wrapper.temperature == 0.2
    assert llm_wrapper.max_tokens == 2048

    model = llm_wrapper.get_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "custom-qwen"
    assert model.openai_api_base == "http://192.168.1.10:9090/v1"
    assert model.temperature == 0.2
    assert model.max_tokens == 2048


def test_openai_model_init():
    llm_wrapper = OpenAIModel(
        model_name="gpt-4o",
        api_key="test-key",
        temperature=0.5,
    )
    assert isinstance(llm_wrapper, BaseLLMModel)
    assert isinstance(llm_wrapper, BaseModel)
    assert llm_wrapper.model_name == "gpt-4o"
    assert llm_wrapper.temperature == 0.5

    dump = llm_wrapper.model_dump()
    assert dump["model_name"] == "gpt-4o"
    assert dump["temperature"] == 0.5

    model = llm_wrapper.get_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o"
    assert model.temperature == 0.5


def test_anthropic_model_init():
    llm_wrapper = AnthropicModel(
        model_name="claude-3-5-haiku-latest",
        api_key="test-anthropic-key",
        temperature=0.3,
    )
    assert isinstance(llm_wrapper, BaseLLMModel)
    assert isinstance(llm_wrapper, BaseModel)
    assert llm_wrapper.model_name == "claude-3-5-haiku-latest"
    assert llm_wrapper.temperature == 0.3

    model = llm_wrapper.get_model()
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-3-5-haiku-latest"
    assert model.temperature == 0.3


def test_aliases():
    assert LlamaCPP is LlamaCPPModel
    assert OpenAI is OpenAIModel
    assert Anthropic is AnthropicModel
    assert get_model is get_llm


def test_get_llm_factory():
    llm_llama = get_llm("llamacpp", model_name="test-llama")
    assert isinstance(llm_llama, ChatOpenAI)

    llm_openai = get_llm("openai", api_key="sk-fake", model_name="gpt-4o-mini")
    assert isinstance(llm_openai, ChatOpenAI)

    llm_anthropic = get_llm("anthropic", api_key="sk-fake-ant", model_name="claude-3-5-sonnet-latest")
    assert isinstance(llm_anthropic, ChatAnthropic)


def test_get_llm_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm("unsupported-provider")


def test_base_model_delegation():
    llm_wrapper = OpenAIModel(api_key="sk-fake", model_name="gpt-4o-mini")
    assert hasattr(llm_wrapper, "bind_tools")
    assert callable(llm_wrapper.bind_tools)
    assert hasattr(llm_wrapper, "with_structured_output")
