#!/usr/bin/env bash

source .env

CONTAINER_IMAGE="${DOCKER_REPO}/pytorch-model-trainer-cpu:latest"
MACHINE_TYPE="n1-standard-4"

gcloud ai custom-jobs create \
    --region=${GCP_REGION} \
    --project=${GCP_PROJECT} \
    --display-name="pytorch-model-trainer-cpu" \
    --worker-pool-spec=machine-type=${MACHINE_TYPE},replica-count=1,container-image-uri=${CONTAINER_IMAGE} \
    --args=train,hello
