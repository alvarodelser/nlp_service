import os
import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

_PROMPT_TEMPLATE = """\
Eres un editor de noticias especializado en movilidad urbana sostenible en España.

Texto completo del artículo:
{text}

Resumen extractivo (contexto):
{extract}

Tu tarea:
1. Escribe un titular conciso y preciso (máximo 15 palabras).
2. Escribe un resumen de exactamente 3 frases que explique el quién, qué y dónde de la noticia.

Formato de respuesta (usa exactamente estas etiquetas):
TITULAR: <titular>
RESUMEN: <resumen en 3 frases>
"""


async def generate_summary(text: str, extract: str) -> tuple[str, str]:
    prompt = _PROMPT_TEMPLATE.format(text=text[:4000], extract=extract)
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        r = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        raw = r.json()["response"].strip()

    return _parse_response(raw)


def _parse_response(raw: str) -> tuple[str, str]:
    headline, summary = "", ""
    for line in raw.splitlines():
        if line.startswith("TITULAR:"):
            headline = line[len("TITULAR:"):].strip()
        elif line.startswith("RESUMEN:"):
            summary = line[len("RESUMEN:"):].strip()
    return headline, summary
