FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK stopwords for sumy
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Pre-pull NLP models so first request is fast (~2 GB total)
# NER: Flair Spanish large (XLM-R + character LM, F1=90.54)
RUN python -c "from flair.models import SequenceTagger; SequenceTagger.load('flair/ner-spanish-large')"
# NLI: hypothesis scoring (nli) + geotagger typing/tie-break
RUN python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='Recognai/bert-base-spanish-wwm-cased-xnli')"

COPY . /app/

# Build the GeoNames gazetteer (HTTPS download from geonames.org, stdlib only, no DB).
RUN python scripts/build_geonames_es.py --out nlp/geotagger/data/geonames_es.tsv

# Explicit guard: fail the build if required data/config files are missing.
RUN test -f /app/nlp/geotagger/data/geonames_es.tsv && \
    test -f /app/config/topics.yaml

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
