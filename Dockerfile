FROM python:3.9-slim-buster
WORKDIR /backend
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
COPY . .
CMD ["gunicorn", "wsgi:app", "-w 2", "-b 0.0.0.0:8080", "-t 30"]

# # Use an official Python runtime as a parent image
# FROM python:3.9-slim-buster

# # Set the working directory in the container
# WORKDIR /app

# # Add the current directory contents into the container at /app
# ADD . /app

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Make port 80 available to the world outside this container
# EXPOSE 8080

# # Run app.py when the container launches
# CMD ["python", "app.py"]