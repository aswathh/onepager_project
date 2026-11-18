# llm/ollama_client.py

# Must match the exact name shown by:  ollama list
MODEL_NAME = "qwen3:8b"

OLLAMA_URL = "http://localhost:11434/api/chat"


def check_ollama():
    """Quick check that Ollama is running and the model exists."""

    import requests

    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10
        )
        response.raise_for_status()
    except Exception as error:
        raise RuntimeError(
            f"Ollama is not reachable: {error}. "
            "Start it with: ollama serve"
        )

    names = [m["name"] for m in response.json().get("models", [])]

    if MODEL_NAME not in names:
        raise RuntimeError(
            f"Model '{MODEL_NAME}' not found. "
            f"Installed models: {names}. "
            "Change MODEL_NAME or run: ollama pull " + MODEL_NAME
        )

    print("Ollama OK, model found:", MODEL_NAME)


if __name__ == "__main__":
    check_ollama()