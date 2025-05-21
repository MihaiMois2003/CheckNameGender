FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Verifică dacă modelele există
RUN ls -la

# Setează variabila de mediu port
ENV PORT=8080

# Rulează aplicația cu gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT app:app