# Train and Deploy R Model

## Summary

Train a model in R on managed notebook and deploy in Vertex AI 


## Workflow Steps

### Setup

1. Setup local environment 
    * install packages via `00_install.R`
2. Setup GCP environment
    * create GCP project
    * authenticate 
    * enable APIs
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



