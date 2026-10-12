#!/usr/bin/env bash
# Deploy the RAGAgent web demo to Cloud Run (builds from source with Cloud Build).
#
#   PROJECT=your-gcp-project ./deploy/cloud_run.sh
#
# Prerequisites: gcloud logged in, billing enabled, and the manuals indexed locally
# (python scripts/fetch_manuals.py --ingest) because index/ is baked into the image.
# The service runs as a dedicated service account with only roles/aiplatform.user,
# so no API keys are involved. Cost guards: max 1 instance, scale to zero, and the
# app-level caps below.
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to your GCP project id}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-ragagent}"
SA_NAME="${SA_NAME:-ragagent-run}"
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

test -d index && ls index/*/meta.json >/dev/null 2>&1 || {
  echo "No manuals indexed in ./index - run: python scripts/fetch_manuals.py --ingest" >&2
  exit 1
}

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com aiplatform.googleapis.com --project "$PROJECT"

if ! gcloud iam service-accounts describe "$SA" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" \
    --display-name "RAGAgent Cloud Run runtime"
fi
gcloud projects add-iam-policy-binding "$PROJECT" --condition=None \
  --member "serviceAccount:${SA}" --role roles/aiplatform.user >/dev/null

gcloud run deploy "$SERVICE" \
  --project "$PROJECT" --region "$REGION" --source . \
  --service-account "$SA" \
  --allow-unauthenticated \
  --min-instances 0 --max-instances 1 --concurrency 8 \
  --cpu 1 --memory 2Gi --timeout 3600 --cpu-boost --session-affinity \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=${PROJECT},GOOGLE_CLOUD_LOCATION=global,RAGAGENT_DAILY_QUESTION_CAP=${RAGAGENT_DAILY_QUESTION_CAP:-300},RAGAGENT_DAILY_UPLOAD_CAP=${RAGAGENT_DAILY_UPLOAD_CAP:-10},RAGAGENT_RATE_PER_MIN=${RAGAGENT_RATE_PER_MIN:-8},RAGAGENT_MAX_TURNS=${RAGAGENT_MAX_TURNS:-30}"

gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format "value(status.url)"
