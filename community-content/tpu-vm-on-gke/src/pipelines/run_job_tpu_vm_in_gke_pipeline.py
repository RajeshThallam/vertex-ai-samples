# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipeline to submit a job on TPU VM in GKE"""

from typing import NamedTuple
from kfp.v2 import dsl
from kfp.v2.dsl import Artifact, Input

from src import config

@dsl.component(
    base_image=config.TPU_IN_GKE_IMAGE
)
def create_tpu_vm(project: str,
                  zone: str,
                  cluster: str,
                  gke_tpu_pool_name: str,
                  tpu_name: str,
                  tpu_type: str,
                  tpu_runtime_version: str
    ):
    """Component to create a TPU VM, join a GKE cluster and 
       register itself as a GKE Node
    """
    import create_tpu_vm

    args = [
        f'',
        f'--zone={zone}', 
        f'--project={project}',
        f'--accelerator_type={tpu_type}', 
        f'--tpu_name={tpu_name}', 
        f'--cluster={cluster}',
        f'--tpu_pool={gke_tpu_pool_name}', 
        f'--runtime_version={tpu_runtime_version}',
    ]
    create_tpu_vm.main(args)

@dsl.component(
    base_image=config.TPU_IN_GKE_IMAGE
)
def delete_tpu_vm(project: str,
                  zone: str,
                  tpu_name: str,
                  pipeline_job_id:str
                 ):
    """Component to delete the TPU VM created prior in the pipeline
    """
    import delete_tpu_vm

    args = [
        f'',
        f'--zone={zone}', 
        f'--project={project}', 
        f'--tpu_name={tpu_name}'
    ]
    delete_tpu_vm.main(args)


@dsl.component(
    base_image=config.TPU_IN_GKE_IMAGE
)
def submit_job_gke(project: str,
                   zone: str,
                   cluster: str,
                   job_spec_path: Input[Artifact]
                  ) -> NamedTuple("Outputs", [("status", str)]):
    """Component to submit job as a workload in GKE. 
       This runs containers directly on the TPU VM through GKE.
    """
    import submit_job_gke
    
    args = [
        f'',
        f'--zone={zone}', 
        f'--project={project}',
        f'--cluster={cluster}',
        f'--job_spec_path={job_spec_path.path}'
    ]
    status = submit_job_gke.main(args)
    return (status,)


@dsl.pipeline(name='pipeline-tpu-in-gke')
def tpu_in_gke_pipeline(
    project: str,
    zone: str,
    cluster: str,
    job_spec_path: str,
    gke_tpu_pool_name: str = 'tpu-pool',
    tpu_name: str = 'gke-tpu',
    tpu_type: str = 'v3-8',
    tpu_runtime_version: str = 'tpu-vm-base',
):
    # create TPU VM and register with the GKE cluster
    create_tpu_vm_op = (
        create_tpu_vm(
            project=project,
            zone=zone,
            cluster=cluster,
            gke_tpu_pool_name=gke_tpu_pool_name,
            tpu_name=tpu_name, 
            tpu_type=tpu_type, 
            tpu_runtime_version=tpu_runtime_version
        )
        .set_display_name("Create and register TPU VM with GKE")
        .set_retry(num_retries=3, backoff_duration="2m", backoff_factor=2)
    )

    # import job spec to deploy on GKE cluster
    tpu_job_spec = (
        dsl.importer(
            artifact_uri=job_spec_path,
            artifact_class=dsl.Artifact,
            reimport=True)
        .set_display_name('Fetch job spec')
        .after(create_tpu_vm_op)
    )
    
    # submit the job to run on TPU VM in GKE
    gke_submit_job_tpu_op = (
        submit_job_gke(
            project=project,
            zone=zone,
            cluster=cluster,
            job_spec_path=tpu_job_spec.output
        )
        .set_display_name("Submit job to TPU on GKE")
    )

    # delete TPU VM
    delete_tpu_vm_op = (
        delete_tpu_vm(
            project=project,
            zone=zone,
            tpu_name=tpu_name,
            pipeline_job_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER
        )
        .set_display_name("Delete TPU VM")
        .after(gke_submit_job_tpu_op)
    )