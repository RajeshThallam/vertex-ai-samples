# 02_train.R --------------------------------------------------------------

## load packages 
library(here)
library(randomForest)

library(googleCloudStorageR)
library(gargle)

## set default project and bucket via environment in .Renviron 
project_id <- Sys.getenv("GCP_PROJECT_ID")
bucket <- Sys.getenv("GCS_BUCKET") # TODO @justinjm - switch this and setup to GCS_DEFAULT_BUCKET
email <- Sys.getenv("GARGLE_AUTH_EMAIL")


## train model
model <- randomForest(Species ~ ., data = iris)

## save model
save(model, file = here("container","model.RData"))

## upload model file to GCS for deployment/serving
## https://cran.r-project.org/web/packages/googleCloudStorageR/vignettes/googleCloudStorageR.html

### Fetch token. See: https://developers.google.com/identity/protocols/oauth2/scopes
scope <- c("https://www.googleapis.com/auth/cloud-platform")
token <- token_fetch(scopes = scope,
                     email = email)

### Pass your token to gcs_auth
gcs_auth(token = token)

### execute upload 
gcs_upload(file = here("container","model.RData"),
           name = "model.RData",
           bucket = bucket)