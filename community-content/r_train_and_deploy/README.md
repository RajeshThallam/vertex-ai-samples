# Train and Deploy R Model


## Summary

Train a model in R on managed notebook and deploy in Vertex AI 

See related notebook [here](https://github.com/RajeshThallam/vertex-ai-labs/tree/main/06-vertex-train-deploy-r-model )


## Pre-Work 


2. Setup GCP environment
    * create GCP project 
    * enable billing 
    * enable APIs
3. create notebook instance for model 


### 1.1 Set up your Google Cloud project

The following steps are required, regardless of your notebook environment.

1. [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager). When you first create an account, you get a $300 free credit towards your compute/storage costs.
2. [Make sure that billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).
3. Enable the Vertex AI API and Compute Engine API.
  * [Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com)
  * [Notebook API](https://console.cloud.google.com/flows/enableapi?apiid=notebooks.googleapis.com)
  * [Cloud Storage API](https://console.cloud.google.com/flows/enableapi?apiid=storage.googleapis.com)
  * [Container Registry API](https://console.cloud.google.com/flows/enableapi?apiid=containerregistry.googleapis.com)
4. If you are running this notebook locally, you will need to install the [Cloud SDK](https://cloud.google.com/sdk).

### 1.2 Create Notebook Instance for model training

```sh
gcloud notebooks instances create r-notebook-instance \
    --vm-image-project=deeplearning-platform-release \
    --vm-image-family=r-4-0-cpu-experimental-notebooks \
    --machine-type=n1-standard-4 \
    --location=us-central1-a \
    --boot-disk-size=100 \
    --network=default
```


### Setup local environment 

* install packages via `00_install.R`
* authenticate 
  * create GCS bucket

    
### Training

3. prepare data
    * get data from source ([Bank marketing - Dataset - DataHub - Frictionless Data](https://datahub.io/machine-learning/bank-marketing#data)) and load to GCS 
4. train model 
    * download data to environment
    * save model to GCS 

### Deployment

5. create build image 
6. upload to container registry
7. create model resource from custom serving container 
8. create endpoint 
9. send online prediction request 



