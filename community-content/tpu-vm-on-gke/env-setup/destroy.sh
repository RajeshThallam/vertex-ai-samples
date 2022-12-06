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

# Cleanup the resources created for running a job on TPU VMs in GKE
# USAGE:  ./destroy.sh PROJECT_ID GKE_CLUSTER NAME_PREFIX [ZONE=us-central1-b]
# ./destroy.sh your-project-id gke-tpu-cluster gke-tpu us-central1-b

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
  echo 'USAGE:  ./destroy.sh PROJECT_ID GKE_CLUSTER NAME_PREFIX [ZONE=us-central1-b]'
  exit 1
fi

TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`

# Set script constants

PROJECT_ID=${1}
GKE_CLUSTER=${2}
NAME_PREFIX=${3:-$PROJECT_ID}
ZONE=${4:-us-central1-b}
REGION="${ZONE::-2}"

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

GCS_BUCKET_NAME=${NAME_PREFIX}-${PROJECT_ID}
NB_INSTANCE_NAME=nb-${NAME_PREFIX}
NB_ZONE='us-central1-a'
DOCKER_ARTIFACT_REPO="tpu-vm-on-gke"

echo ${TIMESTAMP} INFO: Deleting following resources
echo ${TIMESTAMP} INFO:     GKE cluster                      = ${GKE_CLUSTER}
echo ${TIMESTAMP} INFO:     Cloud Storage bucket             = ${GCS_BUCKET_NAME}
echo ${TIMESTAMP} INFO:     Vertex AI Workbench Notebook     = ${NB_INSTANCE_NAME}
echo ${TIMESTAMP} INFO:     Docker repo in Artifact Registry = ${REGION}-docker.pkg.dev/${PROJECT_ID}/${DOCKER_ARTIFACT_REPO}

# Delete GKE cluster if exists
CHECK_CLUSTER=$(is_cluster_exists ${GKE_CLUSTER} ${ZONE})

if [ ${CHECK_CLUSTER} -eq 0 ]; then
  echo ${TIMESTAMP} INFO: Deleting GKE cluster ${GKE_CLUSTER} from ${PROJECT_ID} in zone ${ZONE}
  gcloud container clusters delete ${GKE_CLUSTER} --zone=${ZONE}
else
  echo ${TIMESTAMP} INFO: GKE cluster ${GKE_CLUSTER} not found in ${PROJECT_ID} in zone ${ZONE}
fi

# Deleting Google Cloud Storage Bucket
if gcloud storage buckets list gs://${GCS_BUCKET_NAME} &> /dev/null; then
  echo ${TIMESTAMP} INFO: Deleting bucket gs://${GCS_BUCKET_NAME}
  gcloud storage rm --recursive gs://${GCS_BUCKET_NAME}
fi

# Deleting notebook instance
CHECK_NB=$(is_notebook_exists ${NB_INSTANCE_NAME} ${NB_ZONE})

if [ ${CHECK_NB} -eq 0 ]; then
  echo ${TIMESTAMP} INFO: Deleting Vertex AI Workbench Notebook instance ${NB_INSTANCE_NAME}
  gcloud notebooks instances delete ${NB_INSTANCE_NAME} --location=${NB_ZONE}
fi

# Delete Docker repo in Artifact Repository
CHECK_REPO=$(is_registry_exists ${DOCKER_ARTIFACT_REPO} ${REGION})

if [ ${CHECK_REPO} -eq 0 ]; then
  echo ${TIMESTAMP} INFO: Deleting Docker repository ${DOCKER_ARTIFACT_REPO} in Artifact Registry
  gcloud artifacts repositories delete ${DOCKER_ARTIFACT_REPO} --location=${REGION}
fi

echo ${TIMESTAMP} INFO: Deleted following resources
echo ${TIMESTAMP} INFO:     GKE cluster                      = ${GKE_CLUSTER}
echo ${TIMESTAMP} INFO:     Cloud Storage bucket             = ${GCS_BUCKET_NAME}
echo ${TIMESTAMP} INFO:     Vertex AI Workbench Notebook     = ${NB_INSTANCE_NAME}
echo ${TIMESTAMP} INFO:     Docker repo in Artifact Registry = ${DOCKER_ARTIFACT_REPO}