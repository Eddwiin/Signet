from functools import lru_cache
from pathlib import Path
from openai import OpenAI
from signet.config import get_settings
import gradio as gr
import glob
import tiktoken

BASE_DIR = Path(__file__).resolve().parent
@lru_cache
def get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key.get_secret_value())


def _text(content) -> str:
    """Gradio 6 peut stocker le contenu sous forme de liste de blocs : on garde le texte."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def _read_text_utf8(path: Path) -> str:
    """Lit un fichier en UTF-8. S'il est en cp1252, le convertit en UTF-8 sur le disque."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1252")
        path.write_text(text, encoding="utf-8")
        print(f"Converti en UTF-8: {path.name}")
        return text

def read_data_file():
    # files = glob.glob(str(BASE_DIR / "data" / "*"))
    files = [p for p in (BASE_DIR / "data").glob("*") if p.is_file()]
    print(f"Found {len(files)} files in the knowledge base")
    contents = []

    for path in files:
        try:
            contents.append(_read_text_utf8(path))
        except OSError as e:
            print(f"File not accessible: {path.name} ({e}")

    knowledge_base = "\n\n".join(contents)
    print(f"Total characters in knowledge base: {len(knowledge_base):,}")
    print(display_number_token(knowledge_base))
    return knowledge_base

def display_number_token(entire_knowledge_base):
    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode(entire_knowledge_base)
    token_count = len(tokens)
    print(f"Total tokens for {get_settings().openai_model}: {token_count:,}")


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
