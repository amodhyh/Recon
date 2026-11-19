from recon.config import get_llm
import logging


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )

def main():
    chat_model = get_llm()
    res = chat_model.invoke("tell me about WSO2")
    print(res.content)


if __name__ == "__main__":
    main()
