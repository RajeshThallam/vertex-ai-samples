#!/bin/bash
# Copyright 2022 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Provision the prerequisites for running a job on TPU VMs in GKE 
# using a Vertex AI Pipeline 
# USAGE:  ./install.sh PROJECT_ID GKE_CLUSTER NAME_PREFIX [ZONE=us-central1-b]
# ./install.sh your-project-id gke-tpu-cluster gke-tpu us-central1-b

# Set up a global error handler
err_handler() {
    echo "Error on line: $1"
    echo "Caused by: $2"
    echo "That returned exit status: $3"
    echo "Aborting..."
    exit $3
}

trap 'err_handler "$LINENO" "$BASH_COMMAND" "$?"' ERR

is_cluster_exists() {
    cluster_name=$1
    cluster_zone=$2
    check_cluster=$(gcloud container clusters list \
                      --region ${cluster_zone} \
                      --format="value(name)" \
                      --filter="name:${cluster_name}" \
                      --quiet)
    if [ "${check_cluster}" == "${cluster_name}" ]; then
      echo 0
    else
      echo 1
    fi
}

is_registry_exists() {
    repo_name=$1
    repo_region=$2
    check_repo=$(gcloud artifacts repositories list \
                   --location=${repo_region} \
                   --format="value(name)" \
                   --filter="name~"${repo_name})
    if [ "${check_repo}" == "${repo_name}" ]; then
      echo 0
    else
      echo 1
    fi
}

is_notebook_exists() {
    nb_name=$1
    nb_region=$2
    check_nb=$(gcloud notebooks instances list \
                   --location=${nb_region} \
                   --format="value(name)" \
                   --filter="name~"${nb_name})
    if [ "${check_nb}" == "${nb_name}" ]; then
      echo 0
    else
      echo 1
    fi
}



# Check command line parameters
if [[ $# < 3 ]]; then
  echo 'USAGE:  ./install.sh PROJECT_ID GKE_CLUSTER NAME_PREFIZ [ZONE=us-central1-b]'
  exit 1
fi

TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`

# Set script constants

PROJECT_ID=${1}
GKE_CLUSTER=${2}
NAME_PREFIX=${3:-$PROJECT_ID}
ZONE=${4:-us-central1-b}
REGION="${ZONE::-2}"

GCS_BUCKET_NAME=${NAME_PREFIX}-${PROJECT_ID}
NB_INSTANCE_NAME=nb-${NAME_PREFIX}
NB_ZONE='us-central1-a'
DOCKER_ARTIFACT_REPO="tpu-vm-on-gke"
IMAGE_NAME="tpu-vm-on-gke-base"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${DOCKER_ARTIFACT_REPO}/${IMAGE_NAME}"

echo ${TIMESTAMP} INFO: Creating following resources in PROJECT_ID=${PROJECT_ID}, ZONE=${ZONE} and REGION=${REGION}
echo ${TIMESTAMP} INFO:     GKE cluster                      = ${GKE_CLUSTER}
echo ${TIMESTAMP} INFO:     Cloud Storage bucket             = ${GCS_BUCKET_NAME}
echo ${TIMESTAMP} INFO:     Vertex AI Workbench Notebook     = ${NB_INSTANCE_NAME}
echo ${TIMESTAMP} INFO:     Docker repo in Artifact Registry = ${DOCKER_ARTIFACT_REPO}
echo ${TIMESTAMP} INFO:     Base image in the docker repo    = ${IMAGE_URI}

# Set project
echo ${TIMESTAMP} INFO: Setting the project to: ${PROJECT_ID}
gcloud config set project ${PROJECT_ID}

# Enable services
echo ${TIMESTAMP} INFO: Enabling required services

gcloud services enable \
cloudbuild.googleapis.com \
container.googleapis.com \
cloudresourcemanager.googleapis.com \
iam.googleapis.com \
containerregistry.googleapis.com \
containeranalysis.googleapis.com \
artifactregistry.googleapis.com \
tpu.googleapis.com \
notebooks.googleapis.com \
aiplatform.googleapis.com

echo ${TIMESTAMP} INFO: Required services enabled

# Create GKE cluster if does not exists
echo ${TIMESTAMP} INFO: Validating GKE cluster ${GKE_CLUSTER}

CHECK_CLUSTER=$(is_cluster_exists ${GKE_CLUSTER} ${ZONE})

if [ ${CHECK_CLUSTER} -eq 0 ]; then
  echo ${TIMESTAMP} INFO: GKE cluster ${GKE_CLUSTER} already exists in ${PROJECT_ID} in zone ${ZONE}
else
  echo ${TIMESTAMP} INFO: Creating GKE cluster ${GKE_CLUSTER} in ${PROJECT_ID} in zone ${ZONE}
  gcloud container clusters create ${GKE_CLUSTER} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --cluster-version="1.22" \
    --no-enable-shielded-nodes \
    --no-enable-ip-alias \
    --num-nodes 1
  CHECK_CLUSTER=$(is_cluster_exists ${GKE_CLUSTER} ${ZONE})
  if [ ${CHECK_CLUSTER} -eq 0 ]; then
     echo ${TIMESTAMP} INFO: Created GKE cluster ${GKE_CLUSTER}
  else
     echo ${TIMESTAMP} ERROR: Failed to GKE cluster ${GKE_CLUSTER}
     exit
  fi
fi

# Create Google Cloud Storage Bucket
if ! gcloud storage buckets list gs://${GCS_BUCKET_NAME} &> /dev/null; then
  echo ${TIMESTAMP} INFO: Creating bucket gs://${GCS_BUCKET_NAME}
  gcloud storage buckets create gs://${GCS_BUCKET_NAME} --location=${REGION}
fi

# Create Notebook instance
CHECK_NB=$(is_notebook_exists ${NB_INSTANCE_NAME} ${NB_ZONE})

if [ ${CHECK_NB} -eq 0 ]; then
  echo ${TIMESTAMP} INFO: Vertex AI Workbench Notebook instance ${NB_INSTANCE_NAME} already exists in ${PROJECT_ID} in zone ${NB_ZONE}
else
  echo ${TIMESTAMP} INFO: Creating Vertex AI Workbench Notebook instance ${NB_INSTANCE_NAME} in ${PROJECT_ID} in zone ${NB_ZONE}
  gcloud notebooks instances create ${NB_INSTANCE_NAME} \
    --machine-type=n1-standard-4 \
    --location=${NB_ZONE} \
    --vm-image-project="deeplearning-platform-release" \
    --vm-image-family="common-cpu"
fi

# Build base image 
CHECK_REPO=$(is_registry_exists ${DOCKER_ARTIFACT_REPO} ${REGION})

if [ ${CHECK_REPO} -eq 1 ]; then
  echo ${TIMESTAMP} INFO: Creating a new Docker repository in Artifact Registry to save base images for pipeline

  gcloud artifacts repositories create ${DOCKER_ARTIFACT_REPO} \
    --repository-format=docker \
    --location=${REGION} \
    --description="TPU VM on GKE Docker repository"

  CHECK_REPO=$(is_registry_exists ${DOCKER_ARTIFACT_REPO} ${REGION})
  if [ ${CHECK_REPO} -eq 0 ]; then
     echo ${TIMESTAMP} INFO: Docker repository ${DOCKER_ARTIFACT_REPO} created
  else
     echo ${TIMESTAMP} ERROR: Failed to create Docker repository ${DOCKER_ARTIFACT_REPO} created
     exit
  fi
fi

echo ${TIMESTAMP} INFO: Authenticating to docker repository 
gcloud auth configure-docker {REGION}-docker.pkg.dev --quiet

echo ${TIMESTAMP} INFO: Building and pushing base image to docker repository. This may take up to 30min ...
FILE_LOCATION='.'

gcloud builds submit \
  --region $REGION \
  --config cloudbuild.yaml \
  --substitutions _DOCKERNAME=base,_IMAGE_URI=$IMAGE_URI,_FILE_LOCATION=$FILE_LOCATION \
  --timeout "2h" \
  --quiet

PROJECT_NUMBER=$(gcloud projects list --filter="$(gcloud config get-value project)" --format="value(PROJECT_NUMBER)")
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member "serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role "roles/storage.objectViewer"
  
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member "serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role "roles/storage.objectCreator"

echo ${TIMESTAMP} INFO: Created following resources
echo ${TIMESTAMP} INFO:     GKE cluster                      = ${GKE_CLUSTER}
echo ${TIMESTAMP} INFO:     Cloud Storage bucket             = ${GCS_BUCKET_NAME}
echo ${TIMESTAMP} INFO:     Vertex AI Workbench Notebook     = ${NB_INSTANCE_NAME}
echo ${TIMESTAMP} INFO:     Docker repo in Artifact Registry = ${DOCKER_ARTIFACT_REPO}
echo ${TIMESTAMP} INFO:     Base image in the docker repo    = ${IMAGE_URI}