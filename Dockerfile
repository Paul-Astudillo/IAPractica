# Use an official Python runtime as a parent image
FROM python:3.7.9

# Set the working directory to /usr/src/app
WORKDIR /usr/src/app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y libsasl2-dev libldap2-dev && \
    pip install --no-cache-dir -r requirements.txt

# Bundle app source
COPY . .

# Expose port 8000 and CMD to run the Django server
EXPOSE 8000
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
