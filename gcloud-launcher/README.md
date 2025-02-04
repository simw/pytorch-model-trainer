# Launching jobs in Google Cloud with Vertex AI

Add a .env file with:

```bash
GCP_REGION="<region>"
GCP_PROJECT="<project>"
DOCKER_REPO="<repo>"
```

Run `gcloud auth login`.

Then use the run.sh script:

```bash
uv run cli.py cuda <settings_json_path>
```
