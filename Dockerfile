# Use an official Python runtime as a parent image
FROM python:3.11-slim

RUN --mount=type=cache,target=/var/cache/apt apt-get update && apt-get upgrade -y && apt-get install -y build-essential && apt-get clean all 

RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY llamaindex_simple_graph_rag.py app.py
COPY lib/ lib
CMD ["python", "app.py"]
