FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

RUN python -m py_compile app_main.py && \
    python -m py_compile train_all_horizons.py

EXPOSE 8502

CMD ["python", "-m", "streamlit", "run", "app_main.py", "--server.address", "0.0.0.0", "--server.port", "8502"]
