#!/bin/sh

# https://stackoverflow.com/a/75629058/1839099
DOCKER_BUILDKIT=0 docker build -t gadicc/upsample-api --build-arg OPTIMIZE=1 .
