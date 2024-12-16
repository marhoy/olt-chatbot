#!/bin/bash

appctl job push --yes -a olympiatop -e test -n indexing -d Dockerfile.job && \
appctl service push --yes -a olympiatop -e test -n webapp -d Dockerfile.web
