FROM python:3.9

WORKDIR /app

RUN pip install --no-cache-dir fastapi[standard] uvicorn transformers torch accelerate

COPY . /app

EXPOSE 8000

CMD ["fastapi", "run", "app.py", "--port", "8000"]
