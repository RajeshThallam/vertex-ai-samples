# setup.R -----------------------------------------------------------------

## Setup GCP project

# https://ssh.cloud.google.com/cloudshell/editor

# export PROJECT=$DEVSHELL_PROJECT_ID
# 
# gcloud services enable --project $PROJECT notebooks.googleapis.com
# gcloud services enable --project $PROJECT aiplatform.googleapis.com
# gcloud services enable --project $PROJECT artifactregistry.googleapis.com

### Not needed ###
# gcloud services enable --project $PROJECT monitoring.googleapis.com
# gcloud services enable --project $PROJECT logging.googleapis.com
# gcloud services enable --project $PROJECT compute.googleapis.com
# gcloud services enable --project $PROJECT cloudbuild.googleapis.com
# gcloud services enable --project $PROJECT container.googleapis.com
# gcloud services enable --project $PROJECT bigquery.googleapis.com

### Create GCS bucket 

## set default project and bucket via environment in .Renviron 
project_id <- Sys.getenv("GCP_PROJECT_ID")
bucket <- Sys.getenv("GCS_BUCKET")
email <- Sys.getenv("GARGLE_AUTH_EMAIL")

# load packages 
library(googleCloudStorageR)
library(gargle)

## Fetch token. See: https://developers.google.com/identity/protocols/oauth2/scopes
scope <- c("https://www.googleapis.com/auth/cloud-platform")
token <- token_fetch(scopes = scope,
                     email = email)

## Pass your token to gcs_auth
gcs_auth(token = token)

## create bucket 
gcs_create_bucket(
  name = bucket,
  projectId = project_id,
  location = "us-central1",
  storageClass = "REGIONAL")

## check again to confirm bucket creation successful 
gcs_get_bucket(bucket = bucket)