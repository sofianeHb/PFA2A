FROM python:3.9-slim

WORKDIR /app

COPY ./app /app

RUN pip install --upgrade pip
RUN pip install fastapi uvicorn keras mlflow

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
