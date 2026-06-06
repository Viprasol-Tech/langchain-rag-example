# Part of LangChain RAG Example by Viprasol Tech Private Limited (https://viprasol.com).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples

RUN pip install --no-cache-dir -e .

# Runs the fully offline RAG demo.
CMD ["python", "examples/demo.py"]
