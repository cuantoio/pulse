# Use a base image that supports Python
FROM python:3.9-slim-buster

# Install Nginx
RUN apt-get update && apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/* && \
    echo "\ndaemon off;" >> /etc/nginx/nginx.conf && \
    rm /etc/nginx/sites-enabled/default

# Setup the working directory
WORKDIR /backend

# Copy the requirements and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Nginx configuration template
COPY nginx.conf /etc/nginx/sites-available/backend
RUN ln -s /etc/nginx/sites-available/backend /etc/nginx/sites-enabled/

# Expose ports for Nginx and Gunicorn
EXPOSE 80 8080

# Copy the rest of the application
COPY . .

# Setup the command to start Nginx and Gunicorn
CMD service nginx start && gunicorn wsgi:app -w 2 -b 0.0.0.0:8080 -t 30
