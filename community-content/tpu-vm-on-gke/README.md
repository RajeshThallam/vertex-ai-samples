# Cloud TPU VMs in GKE

This code sample with notebook describes the procedure to run a Vertex AI Pipeline that creates a Cloud TPU VM and automatically join a target GKE cluster and register itself as a GKE Node, allowing you to run containers directly on the TPU VM through GKE. The pipeline supports TensorFlow, PyTorch and JAX and shows example workloads to run on TPU devices and TPU Pods.

## High Level Architecture/Flow

[TO BE ADDED]

## Repository structure


```
.
├── README.md                                           # Start here
├── Dockerfile.base                                     # Dockerfile with dependencies 
├── cloudbuild.yaml
├── env-setup/                                          # Scripts to install and tear down environment
├── src/
│   ├── config.py
│   ├── gke_tpu_vm/                                     # Python modules used by the Pipeline to interact with GKE and Cloud TPU
│   └── pipelines/                                      # Vertex AI Pipeline definition to create TPU VM in GKE, run the job and delete the TPU VM
└── quickstart-tpu-vm-in-gke-vertex-ai-pipeline.ipynb   # Quickstart notebook to run the pipeline including workload/job spec examples
```

## Provisioning an environment

### Environment Requirements

- All services should be provisioned in the same project and the same compute region
- A Google Kubernetes Engine (GKE) cluster located in the [same zone](https://cloud.google.com/tpu/docs/regions-zones) as where you expect to use TPU VM to run jobs
- A Cloud Storage bucket located in the same region as GKE cluster, TPU VM and Vertex AI services. The bucket is used for managing artifacts created by the pipeline.
- Vertex AI Workbench User-managed Notebook instance: as a development environment to customize pipelines and submit and analyze pipeline runs.
- Docker repository in Google Artifact Registry to manage the base image required to run the pipeline
- Vertex AI Pipelines will use the default Compute Engine service account. Following role settings are required to run the pipeline:
   - `roles/storage.objectViewer`
   - `roles/storage.objectCreator`

The repo includes a [set up script](./env-setup/install.sh) to build the environment required for the pipeline. The set up builds the environment as follows:

- [ ] Enable APIs
- [ ] Create a zonal GKE cluster
- [ ] Create a regional Cloud Storage bucket
- [ ] Create a Vertex AI Workbench Notebook instance
- [ ] Create a Docker repository in Cloud Artifact Registry
- [ ] Submit a Cloud Build job to build and push the base image required to run the pipeline
- [ ] Add roles to default Compute Engine service account to use with pipelines

A few things to note:
1. You need to be a project owner to set up the environment. 
2. You will be using [Cloud Shell](https://cloud.google.com/shell/docs/using-cloud-shell) to start and monitor the setup process.
3. The notebook instance is created in `us-central1-a` zone. The notebook instance can be used for running jobs in other region.

Click on the below link to navigate to Cloud Shell and clone the repo.

<a href="https://console.cloud.google.com/cloudshell/open?git_repo=https://github.com/RajeshThallam/vertex-ai-samples&cloudshell_git_branch=tpu-in-gke&cloudshell_workspace=community-content/tpu-vm-on-gke&tutorial=README.md">
    <img alt="Open in Cloud Shell" src="http://gstatic.com/cloudssh/images/open-btn.png">
</a>

### Run environment setup script

Set the below environment variables to reflect your environment. The set up script will attempt to create new resources so make sure that the resources with the specified names do not already exist.

```
export PROJECT_ID=<YOUR PROJECT ID>
export GKE_CLUSTER_NAME=<YOUR GKE CLUSTER NAME>
export NAME_PREFIX=<NAME PREFIX TO ADD TO RESOURCES CREATED>
export ZONE=<YOUR ZONE>
```

Start installation script. This step may take a few minutes.

```
cd vertex-ai-samples/community-content/tpu-vm-in-gke-pipeline/
./env-setup/install.sh $PROJECT_ID $GKE_CLUSTER_NAME $NAME_PREFIX $ZONE
```


## Configuring Vertex Workbench

The environment set up has an instance of Vertex AI Workbench which is used as a development/experimentation environment to customize, start, and analyze pipeline runs. 

Connect to JupyterLab on your Vertex AI Workbench instance and start a JupyterLab terminal.

From the JupyterLab terminal, clone the repo and change to working directory.


```
git clone https://github.com/GoogleCloudPlatform/vertex-ai-samples.git
cd vertex-ai-samples/community-content/tpu-vm-in-gke-pipeline/
```

You are now ready to walk through the [quickstart notebook](./quickstart-tpu-vm-in-gke-vertex-ai-pipeline.ipynb) that demonstrate how to run and customize pipelines. The notebook has examples for running TensorFlow, PyTorch and JAX workloads on TPU devices and TPU Pods.

## Clean up

If you want to remove the resource created for the demo execute the following command from Cloud Shell.

```
cd vertex-ai-samples/community-content/tpu-vm-in-gke-pipeline/
./env-setup/destroy.sh $PROJECT_ID $GKE_CLUSTER_NAME $NAME_PREFIX $ZONE
```

The clean up script removes following resources:

- GKE Cluster
- Cloud Storage bucket
- Vertex AI Workbench Notebook instance
- Docker repository on Artifact Registry
