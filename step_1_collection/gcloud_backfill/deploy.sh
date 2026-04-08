#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Deploy the historical-backfill Cloud Run Job + Cloud Scheduler trigger.
#
# Prerequisites:
#   - gcloud CLI installed & authenticated  (gcloud auth login)
#   - APIs enabled: Cloud Run, Cloud Build, Artifact Registry, Cloud Scheduler
#     gcloud services enable \
#       run.googleapis.com \
#       cloudbuild.googleapis.com \
#       artifactregistry.googleapis.com \
#       cloudscheduler.googleapis.com
#
# Required env-vars:
#   GCP_PROJECT_ID   – your GCP project id
#
# Optional env-vars (defaults shown):
#   GCP_REGION       – europe-west1
#   GCS_BUCKET       – exam-project-backfill
#   ARPA_APP_TOKEN   – (empty, but recommended for higher Socrata rate limits)
#   DAYS_PER_BATCH   – 20  (days per hourly run)
#   BACKFILL_START   – 2016-04-08
#   BACKFILL_END     – 2025-04-07
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Export GCP_PROJECT_ID before running this script}"
REGION="${GCP_REGION:-europe-west1}"
BUCKET="${GCS_BUCKET:-exam-project-backfill}"
JOB_NAME="backfill-historical"
SCHEDULER_NAME="trigger-backfill"
REPO_NAME="exam-project"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${JOB_NAME}:latest"

DAYS="${DAYS_PER_BATCH:-20}"
START="${BACKFILL_START:-2024-01-01}"
END="${BACKFILL_END:-2025-04-07}"
APP_TOKEN="${ARPA_APP_TOKEN:-}"

echo "============================================================"
echo "  Historical Backfill — GCloud Deploy"
echo "============================================================"
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Bucket:   gs://${BUCKET}"
echo "  Image:    ${IMAGE}"
echo "  Range:    ${START} -> ${END}"
echo "  Batch:    ${DAYS} days/run"
echo "============================================================"
echo ""

# -- 1. GCS bucket (idempotent) -----------------------------------------------
echo ">>> Creating GCS bucket..."
gcloud storage buckets create "gs://${BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --uniform-bucket-level-access \
  2>/dev/null || echo "    (bucket already exists)"

# -- 2. Artifact Registry repo (idempotent) -----------------------------------
echo ">>> Creating Artifact Registry repository..."
gcloud artifacts repositories create "${REPO_NAME}" \
  --project="${PROJECT_ID}" \
  --repository-format=docker \
  --location="${REGION}" \
  2>/dev/null || echo "    (repository already exists)"

# -- 3. Build & push Docker image ---------------------------------------------
echo ">>> Building and pushing Docker image..."
gcloud builds submit \
  --project="${PROJECT_ID}" \
  --tag "${IMAGE}" \
  --quiet

# -- 4. Cloud Run Job (create or update) --------------------------------------
echo ">>> Configuring Cloud Run Job..."

ENV_VARS="GCS_BUCKET=${BUCKET}"
ENV_VARS="${ENV_VARS},DAYS_PER_BATCH=${DAYS}"
ENV_VARS="${ENV_VARS},BACKFILL_START=${START}"
ENV_VARS="${ENV_VARS},BACKFILL_END=${END}"
if [ -n "${APP_TOKEN}" ]; then
  ENV_VARS="${ENV_VARS},ARPA_APP_TOKEN=${APP_TOKEN}"
fi

if gcloud run jobs describe "${JOB_NAME}" \
      --project="${PROJECT_ID}" --region="${REGION}" &>/dev/null; then
  gcloud run jobs update "${JOB_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --image="${IMAGE}" \
    --task-timeout=21600 \
    --max-retries=2 \
    --set-env-vars="${ENV_VARS}" \
    --memory=2Gi \
    --cpu=1
else
  gcloud run jobs create "${JOB_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --image="${IMAGE}" \
    --task-timeout=21600 \
    --max-retries=2 \
    --set-env-vars="${ENV_VARS}" \
    --memory=2Gi \
    --cpu=1
fi

# -- 5. Cloud Scheduler (every hour) -------------------------------------------
echo ">>> Configuring Cloud Scheduler..."

# Default Compute Engine SA — adjust if you use a custom SA.
SA_EMAIL="${PROJECT_ID}@appspot.gserviceaccount.com"
JOB_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run"

if gcloud scheduler jobs describe "${SCHEDULER_NAME}" \
      --project="${PROJECT_ID}" --location="${REGION}" &>/dev/null; then
  gcloud scheduler jobs update http "${SCHEDULER_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="0 * * * *" \
    --time-zone="Europe/Rome" \
    --uri="${JOB_URI}" \
    --http-method=POST \
    --oauth-service-account-email="${SA_EMAIL}"
else
  gcloud scheduler jobs create http "${SCHEDULER_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="0 * * * *" \
    --time-zone="Europe/Rome" \
    --uri="${JOB_URI}" \
    --http-method=POST \
    --oauth-service-account-email="${SA_EMAIL}"
fi

# -- Done ---------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Deploy complete!"
echo "============================================================"
echo ""
echo "Useful commands:"
echo ""
echo "  # Run the job manually right now"
echo "  gcloud run jobs execute ${JOB_NAME} --project=${PROJECT_ID} --region=${REGION}"
echo ""
echo "  # Stream live logs"
echo "  gcloud logging read \\"
echo "    'resource.type=\"cloud_run_job\" resource.labels.job_name=\"${JOB_NAME}\"' \\"
echo "    --project=${PROJECT_ID} --limit=50 --format='table(timestamp,textPayload)'"
echo ""
echo "  # Inspect checkpoint"
echo "  gcloud storage cat gs://${BUCKET}/checkpoint/backfill_state.json"
echo ""
echo "  # Pause / resume scheduler"
echo "  gcloud scheduler jobs pause  ${SCHEDULER_NAME} --project=${PROJECT_ID} --location=${REGION}"
echo "  gcloud scheduler jobs resume ${SCHEDULER_NAME} --project=${PROJECT_ID} --location=${REGION}"
echo ""
