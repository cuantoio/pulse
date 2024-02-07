FROM python:3.9-slim-buster
# Install Nginx
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

# Configure Nginx
COPY nginx.conf /etc/nginx/sites-available/default

WORKDIR /backend

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80  # Nginx listens on port 80

CMD ["sh", "-c", "service nginx start && gunicorn wsgi:app -w 2 -b unix:/tmp/gunicorn.sock"]