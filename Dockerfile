FROM python:3.9-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y curl

# Installer pip requirements
COPY requirements_deployement.txt .
RUN pip install --no-cache-dir -r requirements_deployement.txt

# Copier l'app
COPY . /app
WORKDIR /app

# Installer Ngrok
RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc \
 && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list \
 && apt-get update && apt-get install -y ngrok

EXPOSE 8000

CMD ngrok authtoken $NGROK_AUTH_TOKEN && \
    mlflow ui --host 0.0.0.0 --port 5000 & \
    uvicorn app:app --host 0.0.0.0 --port 8000
