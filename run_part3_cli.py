from openai import OpenAI
from agents_part3_local import Head_Agent


def load_key(path="key.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


if __name__ == "__main__":
    api_key = load_key("key.txt")
    client = OpenAI(api_key=api_key)

    head = Head_Agent(openai_client=client, local_vectorstore_path="local_vectorstore.pkl")

    history = ""
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        ans, path = head(q, history=history)
        print(f"\nAssistant ({' -> '.join(path)}):\n{ans}")

        history += f"User: {q}\nAssistant: {ans}\n"