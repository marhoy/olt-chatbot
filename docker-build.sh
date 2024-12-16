#!/bin/bash

GIT_BRANCH=$(git branch --show-current)
GIT_COMMIT=$(git describe --always --dirty)

docker build -t olt_chatbot/job -f Dockerfile.job --build-arg GIT_BRANCH=${GIT_BRANCH} --build-arg GIT_COMMIT=${GIT_COMMIT} . &&\
docker build -t olt_chatbot/web -f Dockerfile.web --build-arg GIT_BRANCH=${GIT_BRANCH} --build-arg GIT_COMMIT=${GIT_COMMIT} .
