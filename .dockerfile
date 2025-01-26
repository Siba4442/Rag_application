# Start from a Python base image
FROM python:3.12-slim

# Install required system dependencies, including sqlite3
RUN apt-get update && apt-get install -y sqlite3

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Define the command to run your Streamlit app
CMD ["streamlit", "run", "streamlit.py"]