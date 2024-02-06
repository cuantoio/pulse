FROM python:3.9-slim-buster
WORKDIR /backend
COPY requirements.txt requirements.txt
COPY nginx.conf nginx.conf
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
COPY . .
CMD ["gunicorn", "wsgi:app", "-w 2", "-b 0.0.0.0:8080", "-t 30"]