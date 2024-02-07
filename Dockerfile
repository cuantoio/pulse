FROM python:3.9-slim-buster
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*
WORKDIR /backend
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY nginx.conf /etc/nginx/sites-available/default
RUN ln -sf /etc/nginx/sites-available/default /etc/nginx/sites-enabled/ && echo "daemon off;" >> /etc/nginx/nginx.conf
EXPOSE 80
CMD service nginx restart && gunicorn wsgi:app -w 2 -b unix:/tmp/gunicorn.sock -t 30
