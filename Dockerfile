# Use Python image
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /backend

# Install NGINX
RUN apt-get update && apt-get install -y nginx

# Copy the requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the custom NGINX config
COPY nginx.conf /etc/nginx/nginx.conf

# Copy the rest of your application
COPY . .

# Setup NGINX to forward requests to Gunicorn
RUN echo "daemon off;" >> /etc/nginx/nginx.conf
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

# Expose ports for NGINX (80 for HTTP, 443 for HTTPS) and Gunicorn
EXPOSE 80 443 8080

# Start both Gunicorn and NGINX
CMD service nginx start && gunicorn wsgi:app -w 2 -b 0.0.0.0:8080 -t 30
