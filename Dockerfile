# MotoMedic rider UI + RAGAgent (Google ADK) backend, for Cloud Run.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY .streamlit ./.streamlit
COPY bike_manual_agent ./bike_manual_agent
COPY rider_ui ./rider_ui
COPY samples ./samples
COPY manuals/sources.json ./manuals/sources.json
# The committed, prebuilt index is copied in: the container never embeds the bundled manuals.
COPY index ./index

RUN useradd --uid 10001 --create-home rider \
    && mkdir -p manuals/uploads \
    && chown -R rider /app/index /app/manuals
USER rider

EXPOSE 8080
CMD ["sh", "-c", "exec streamlit run rider_ui/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
