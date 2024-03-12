#!/bin/bash

LOCAL_PORT=8911
CONFIG_ENV=config.env
DOCKER_OUTPUT=$(pwd)/output-docker

# Remove possible output from previous run
rm -rf ${DOCKER_OUTPUT}
# Create directory and grant other users (the docker container user) write-access
mkdir -m 2777 ${DOCKER_OUTPUT}

# Run the training job
docker run --rm -it -v ${DOCKER_OUTPUT}:/mnt/output --env-file ${CONFIG_ENV} olt_chatbot/job

# Start the API server
docker run --rm -it -p ${LOCAL_PORT}:80 -v ${DOCKER_OUTPUT}:/mnt/input --env-file ${CONFIG_ENV} olt_chatbot/web
