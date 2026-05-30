import numpy as np
from nlp.encoder import encode
from nlp.summarizer.ollama_client import generate_summary


async def summarize(text: str, extract: str, headline: str) -> tuple[str, str, np.ndarray]:
    new_headline, summary = await generate_summary(text, extract)
    if not new_headline:
        new_headline = headline
    embedding_summary = encode(new_headline + " " + summary)
    return new_headline, summary, embedding_summary
