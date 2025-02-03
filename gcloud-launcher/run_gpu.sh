#!/usr/bin/env bash

source .env

CONTAINER_IMAGE="${DOCKER_REPO}/pytorch-model-trainer-cuda:latest"
MACHINE_TYPE="a2-highgpu-1g"

gcloud ai custom-jobs create \
    --region=${GCP_REGION} \
    --project=${GCP_PROJECT} \
    --display-name="pytorch-model-trainer-cuda" \
    --worker-pool-spec=machine-type=${MACHINE_TYPE},replica-count=1,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,container-image-uri=${CONTAINER_IMAGE} \
    --args=train,test
