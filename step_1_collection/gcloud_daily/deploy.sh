#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Deploy the daily-refresh Cloud Run Job + Cloud Scheduler trigger.
#
# Keeps the rolling window (default last 7 days) of ARPA + Open-Meteo data
# fresh on GCS, so the API can compute lag/rolling features at inference.
#
# Shares the bucket, Artifact Registry repo and service account of the
# historical backfill (see gcloud_backfill/deploy.sh).  Run that one at
# least once first to create the underlying infra.
#
# Required env-vars:
#   GCP_PROJECT_ID   – your GCP project id
#
# Optional env-vars (defaults shown):
#   GCP_REGION       – europe-west1
#   GCS_BUCKET       – exam-project-backfill
#   ARPA_APP_TOKEN   – (empty)
#   WINDOW_DAYS      – 7
#   END_OFFSET_DAYS  – 1
# ---------------------------------------------------------------------------
set -euo pipefail

# Always run from the script's own directory so ``gcloud builds submit``
# finds the Dockerfile regardless of where the user invoked the script.
cd "$(dirname "$0")"

PROJECT_ID="${GCP_PROJECT_ID:?Export GCP_PROJECT_ID before running this script}"
REGION="${GCP_REGION:-europe-west1}"
BUCKET="${GCS_BUCKET:-exam-project-backfill}"
JOB_NAME="backfill-daily"
SCHEDULER_NAME="trigger-daily"
REPO_NAME="exam-project"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${JOB_NAME}:latest"

WINDOW="${WINDOW_DAYS:-7}"
END_OFFSET="${END_OFFSET_DAYS:-1}"
APP_TOKEN="${ARPA_APP_TOKEN:-}"

echo "============================================================"
echo "  Daily Refresh — GCloud Deploy"
echo "============================================================"
echo "  Project:   ${PROJECT_ID}"
echo "  Region:    ${REGION}"
echo "  Bucket:    gs://${BUCKET}"
echo "  Image:     ${IMAGE}"
echo "  Window:    ${WINDOW} days, ending today-${END_OFFSET}"
echo "============================================================"
echo ""

# -- 1. Artifact Registry repo (idempotent; created by backfill deploy) -------
echo ">>> Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories create "${REPO_NAME}" \
  --project="${PROJECT_ID}" \
  --repository-format=docker \
  --location="${REGION}" \
  2>/dev/null || echo "    (repository already exists)"

# -- 2. Build & push Docker image ---------------------------------------------
echo ">>> Building and pushing Docker image..."
gcloud builds submit \
  --project="${PROJECT_ID}" \
  --tag "${IMAGE}" \
  --quiet

# -- 3. Cloud Run Job (create or update) --------------------------------------
echo ">>> Configuring Cloud Run Job..."

ENV_VARS="GCS_BUCKET=${BUCKET}"
ENV_VARS="${ENV_VARS},WINDOW_DAYS=${WINDOW}"
ENV_VARS="${ENV_VARS},END_OFFSET_DAYS=${END_OFFSET}"
if [ -n "${APP_TOKEN}" ]; then
  ENV_VARS="${ENV_VARS},ARPA_APP_TOKEN=${APP_TOKEN}"
fi

if gcloud run jobs describe "${JOB_NAME}" \
      --project="${PROJECT_ID}" --region="${REGION}" &>/dev/null; then
  gcloud run jobs update "${JOB_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --image="${IMAGE}" \
    --task-timeout=3600 \
    --max-retries=2 \
    --set-env-vars="${ENV_VARS}" \
    --memory=1Gi \
    --cpu=1
else
  gcloud run jobs create "${JOB_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --image="${IMAGE}" \
    --task-timeout=3600 \
    --max-retries=2 \
    --set-env-vars="${ENV_VARS}" \
    --memory=1Gi \
    --cpu=1
fi

# -- 4. Cloud Scheduler (once per day, 03:00 Europe/Rome) ---------------------
echo ">>> Configuring Cloud Scheduler..."

SA_EMAIL="${PROJECT_ID}@appspot.gserviceaccount.com"
JOB_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run"

if gcloud scheduler jobs describe "${SCHEDULER_NAME}" \
      --project="${PROJECT_ID}" --location="${REGION}" &>/dev/null; then
  gcloud scheduler jobs update http "${SCHEDULER_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="0 3 * * *" \
    --time-zone="Europe/Rome" \
    --uri="${JOB_URI}" \
    --http-method=POST \
    --oauth-service-account-email="${SA_EMAIL}"
else
  gcloud scheduler jobs create http "${SCHEDULER_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="0 3 * * *" \
    --time-zone="Europe/Rome" \
    --uri="${JOB_URI}" \
    --http-method=POST \
    --oauth-service-account-email="${SA_EMAIL}"
fi

echo ""
echo "============================================================"
echo "  Deploy complete!"
echo "============================================================"
echo ""
echo "  # Run the daily job manually"
echo "  gcloud run jobs execute ${JOB_NAME} --project=${PROJECT_ID} --region=${REGION}"
echo ""
echo "  # Pause / resume scheduler"
echo "  gcloud scheduler jobs pause  ${SCHEDULER_NAME} --project=${PROJECT_ID} --location=${REGION}"
echo "  gcloud scheduler jobs resume ${SCHEDULER_NAME} --project=${PROJECT_ID} --location=${REGION}"
echo ""
