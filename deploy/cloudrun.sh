#!/usr/bin/env bash
# Deploy the fraud detection service to Google Cloud Run (or remove it).
#
# Cloud Build builds the image from the Dockerfile in the cloud and stores it in
# Artifact Registry - no local Docker build or manual push needed.
#
# Prerequisites (one-time, see README "Cloud deployment"):
#   gcloud CLI installed, `gcloud init` done, project with billing enabled.
#
# Usage (from the project root):
#   ./deploy/cloudrun.sh deploy     # build + deploy, prints the public URL
#   ./deploy/cloudrun.sh url        # print the URL of the running service
#   ./deploy/cloudrun.sh delete     # delete the service and its images
#
# Settings can be overridden with environment variables, e.g.
#   REGION=us-central1 ./deploy/cloudrun.sh deploy

set -euo pipefail

SERVICE="${SERVICE:-credit-card-fraud}"
REGION="${REGION:-europe-west1}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

if [[ -z "$PROJECT" ]]; then
    echo "No GCP project set. Run 'gcloud init' or 'gcloud config set project <id>'." >&2
    exit 1
fi

cd "$(dirname "$0")/.."

case "${1:-}" in
deploy)
    echo "Enabling required APIs in project $PROJECT (no-op if already enabled) ..."
    gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
        artifactregistry.googleapis.com --project "$PROJECT"

    # In new projects, Cloud Build builds as the Compute Engine default service
    # account, which no longer gets build permissions automatically. Grant it the
    # Cloud Run Builder role (read uploaded source, push image, write logs).
    # Idempotent - re-adding an existing binding is a no-op.
    PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')
    BUILD_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    echo "Granting roles/run.builder to the build service account $BUILD_SA ..."
    gcloud projects add-iam-policy-binding "$PROJECT" \
        --member "serviceAccount:$BUILD_SA" \
        --role roles/run.builder \
        --condition None \
        --quiet >/dev/null

    # Cost / abuse protection:
    #   --max-instances 1   at most one container, however many requests arrive
    #   --concurrency 20    requests handled in parallel by that container
    #   --timeout 10        no request may hold the container longer than 10 s
    #   min instances = 0   (default) scales to zero when idle, so idle costs nothing
    gcloud run deploy "$SERVICE" \
        --source . \
        --project "$PROJECT" \
        --region "$REGION" \
        --port 9696 \
        --memory 512Mi \
        --cpu 1 \
        --max-instances 1 \
        --concurrency 20 \
        --timeout 10 \
        --allow-unauthenticated

    URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
        --format 'value(status.url)')
    echo
    echo "Deployed: $URL"
    echo "  health: curl $URL/health"
    echo "  docs:   $URL/docs"
    echo "  test:   uv run python predict_test.py --url $URL"
    ;;
url)
    gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
        --format 'value(status.url)'
    ;;
delete)
    gcloud run services delete "$SERVICE" --project "$PROJECT" --region "$REGION" --quiet
    # Images built by `--source` deploys live in this Artifact Registry repo.
    if gcloud artifacts repositories describe cloud-run-source-deploy \
        --project "$PROJECT" --location "$REGION" >/dev/null 2>&1; then
        gcloud artifacts repositories delete cloud-run-source-deploy \
            --project "$PROJECT" --location "$REGION" --quiet
    fi
    echo "Deleted service $SERVICE and its images."
    ;;
*)
    sed -n '2,16p' "$0"
    exit 1
    ;;
esac
