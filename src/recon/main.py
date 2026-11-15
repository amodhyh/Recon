from recon.config import get_llm


def main():
    chat_model = get_llm()
    res = chat_model.invoke("tell me about WSO2")
    print(res.content)


if __name__ == "__main__":
    main()
