#!/bin/bash

# Define a variable for the image name
IMAGE_NAME="llama_sim_t1"

TYPE=$1
MODEL=$2
AI_MODEL=$3
IDENT=$4

if [ "$TYPE" == "" ] || [ "$MODEL" == "" ] || [ "$AI_MODEL" == "" ] || [ "$IDENT" == "" ]
then
    echo "Arguments missing"
    echo "Usage: ./build_run.sh <type> <model> <ai_model> <ident>"
    echo "  Examples:"
    echo "       ./build_run.sh together mixtral-together mistralai/Mixtral-8x7B-Instruct-v0.1 AY-yahoo-content-no_sentiment-40"
    echo "       ./build_run.sh ollama codeup ignore AY-yahoo-content-no_sentiment-40"
    exit 1
fi
# Build the Docker image and run if successful
dbuild.sh -t $IMAGE_NAME . && time docker run -it --rm \
    -v "$(pwd)/data:/data" \
    -v "$(pwd)/data_root_nltk:/root/nltk_data" \
    -e TOGETHER_AI_KEY \
    -e OPENAI_API_KEY \
    -e CONF_TYPE="$TYPE" \
    -e CONF_MODEL="$MODEL" \
    -e CONF_AI_MODEL="$AI_MODEL" \
    -e CONF_IDENT="$IDENT" \
 $IMAGE_NAME

sudo chown `whoami` data/*