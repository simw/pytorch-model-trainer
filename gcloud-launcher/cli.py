import json
import os
import sys

import click
from dotenv import load_dotenv
from google.cloud import aiplatform

SETTINGS = {
    "cpu": {
        "container_image": "pytorch-model-trainer-cpu:latest",
        "machine_type": "n1-standard-4",
        "replica_count": 1,
        "accelerator_type": None,
        "accelerator_count": None,
    },
    "cuda": {
        "container_image": "pytorch-model-trainer-cuda:latest",
        "machine_type": "a2-highgpu-1g",
        "replica_count": 1,
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 1,
    },
}


@click.command()
@click.argument("target")
@click.argument("config_file")
def run(target: str, config_file: str) -> None:
    load_dotenv()

    docker_repo = os.getenv("DOCKER_REPO")
    gcp_project = os.getenv("GCP_PROJECT")
    gcp_region = os.getenv("GCP_REGION")

    settings = SETTINGS.get(target)
    if settings is None:
        click.echo(f"Invalid target: {target}")
        sys.exit(1)

    if not os.path.isfile(config_file):
        click.echo(f"File not found: {config_file}")
        sys.exit(1)

    with open(config_file) as file:
        config = json.load(file)

    click.echo(f"Launching custom job with target: {target}")

    aiplatform.init(project=gcp_project, location=gcp_region)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=f"pytorch-model-trainer-{target}",
        container_uri=f"{docker_repo}/{settings['container_image']}",
        staging_bucket="gs://bkt-wrp-d-research-processed-data-c86e08b8",
    )

    job.submit(
        machine_type=settings["machine_type"],
        args=["train", "v1", json.dumps(config)],
        replica_count=settings["replica_count"],
        accelerator_type=settings["accelerator_type"],
        accelerator_count=settings["accelerator_count"],
    )


if __name__ == "__main__":
    run()
