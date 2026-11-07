import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

LLM_NAME = os.getenv("LLM_NAME")
LLM_PROVIDER = os.getenv("LLM_PROVIDER")


def get_llm():
    return init_chat_model(model=LLM_NAME, model_provider=LLM_PROVIDER)


def main():
    chat_model = get_llm()
    res = chat_model.invoke("tell me about WSO2")
    print(res.content)


if __name__ == "__main__":
    main()
