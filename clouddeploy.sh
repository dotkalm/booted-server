#!/bin/bash

# Configuration
PROJECT_ID="your-gcp-project-id"
SERVICE_NAME="edge-detector"
REGION="us-central1"
MEMORY="4Gi"
CPU="2"
MAX_INSTANCES="10"
TIMEOUT="300"

# Build and deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --source . \
  --platform managed \
  --region $REGION \
  --memory $MEMORY \
  --cpu $CPU \
  --timeout $TIMEOUT \
  --max-instances $MAX_INSTANCES \
  --allow-unauthenticated \
  --set-env-vars "ROBOFLOW_API_KEY=your-api-key-here" \
  --project $PROJECT_ID

echo "Deployment complete!"
echo "Service URL: https://$SERVICE_NAME-<hash>-$REGION.run.app"
