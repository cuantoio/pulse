FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

COPY nginx.conf nginx.conf

WORKDIR /backend

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "wsgi:app", "-w 2", "-b 0.0.0.0:8080", "-t 30"]
