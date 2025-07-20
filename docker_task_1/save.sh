#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set default container name
DOCKER_IMAGE_TAG="tia-pegasus-prostate-cancer-biochemical-recurrence-prediction"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/build.sh"

# Get the build information from the Docker image tag
build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")

if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $DOCKER_IMAGE_TAG"
    exit 1
fi

# Format the build information to remove special characters
formatted_build_info=$(echo $build_timestamp | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')

# Set the output filename with timestamp and build information
output_filename="${SCRIPT_DIR}/${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"

# Save the Docker-container image and pigz it
echo "==+=="
echo "Saving the container image as ${output_filename}. This can take a while."

echo ""


docker save "$DOCKER_IMAGE_TAG" | pigz -c > "$output_filename"
echo "Container image saved as ${output_filename}"
echo "==+=="