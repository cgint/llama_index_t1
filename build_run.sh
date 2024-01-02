#!/bin/bash

# Define a variable for the image name
IMAGE_NAME="llama_sim_t1"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Run the Docker container interactively and remove it after it stops
docker run -it --rm -v "$(pwd)/data:/data" $IMAGE_NAME
