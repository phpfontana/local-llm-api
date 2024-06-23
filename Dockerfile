# Use the official Python image from Docker Hub with Python 3.10
FROM python:3.10-bookworm

# Set the working directory in the container
WORKDIR /code

# Copy the file with the requirements to the /backend directory
COPY ./requirements.txt /code/requirements.txt

RUN pip3 install --upgrade pip

# Install the Python dependencies
RUN pip3 install --no-cache-dir -r /code/requirements.txt

# Copy the backend directory to the container
COPY ./app/ /code/app

# Expose the port 80
EXPOSE 80

# Define the command to run the FastAPI app using uvicorn
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "80", "--reload"]