from functools import lru_cache
from openai import OpenAI
from signet.config import get_settings
import gradio as gr

@lru_cache
def get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key.get_secret_value())


def _text(content) -> str:
    """Gradio 6 peut stocker le contenu sous forme de liste de blocs : on garde le texte."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def chat(message: str, history: list[dict]) -> str:
    settings = get_settings()
    messages = [{"role": m["role"], "content": _text(m["content"])} for m in history]
    messages.append({"role": "user", "content": message})
    response = get_openai_client().chat.completions.create(
        model=settings.openai_model,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def open_chat() -> None:
    gr.ChatInterface(chat).launch(inbrowser=True)
