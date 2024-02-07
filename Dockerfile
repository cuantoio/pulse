FROM python:3.9-slim-buster
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*
WORKDIR /backend
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY nginx.conf /etc/nginx/nginx.conf
RUN echo "daemon off;" >> /etc/nginx/nginx.conf
EXPOSE 80
CMD ["sh", "-c", "exec nginx & exec gunicorn wsgi:app -w 2 -b 0.0.0.0:8080 -t 30"]
