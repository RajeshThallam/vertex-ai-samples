# setup.R -----------------------------------------------------------------

## Setup local environment


## Setup GCP project

# https://ssh.cloud.google.com/cloudshell/editor

export PROJECT=$DEVSHELL_PROJECT_ID

gcloud services enable --project $PROJECT notebooks.googleapis.com
gcloud services enable --project $PROJECT aiplatform.googleapis.com
gcloud services enable --project $PROJECT artifactregistry.googleapis.com

# gcloud services enable --project $PROJECT monitoring.googleapis.com
# gcloud services enable --project $PROJECT logging.googleapis.com
# gcloud services enable --project $PROJECT compute.googleapis.com
# gcloud services enable --project $PROJECT cloudbuild.googleapis.com
# gcloud services enable --project $PROJECT container.googleapis.com
# gcloud services enable --project $PROJECT bigquery.googleapis.com

### Create GCS bucket 

