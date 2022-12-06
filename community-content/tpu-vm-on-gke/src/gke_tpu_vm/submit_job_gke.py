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

"""Submit a job to run on TPU VM within the GKE Cluster.

This script is highly experimental. TPU VM GKE nodes may not behave as expected.

Example command:

python submit_job_gke.py \
    --cluster=$USER-cluster\
    --zone=us-central1-a \
    --project=$USER-gke-dev \
    --job_spec_path=/gcs/bucket/training/scripts/job.yaml
"""

import base64
import time
from absl import app
from absl import flags
from typing import Dict, Sequence
import urllib

from google.cloud.container_v1 import ClusterManagerClient
from kubernetes import client, utils
from tempfile import NamedTemporaryFile
import google.auth

def parse_flags(args):
    flags.DEFINE_string('cluster', None, 'Name of GKE cluster')
    flags.DEFINE_string('zone', None, 'GCP zone of TPU and GKE cluster.')
    flags.DEFINE_string('project', None, 'GCP project of TPU and GKE cluster.')
    flags.DEFINE_string('job_spec_path', None, 
                        'Path to the job specification yaml file to submit to the cluster')

    flags.mark_flags_as_required(['cluster', 'zone', 'project', 'job_spec_path'])

def get_config(project, zone, cluster):
    # ref: https://stackoverflow.com/a/68098831/6069517
    # get credentials
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform',])
    credentials.refresh(google.auth.transport.requests.Request())

    # get cluster info
    cluster_manager = ClusterManagerClient(credentials=credentials)
    cluster = cluster_manager.get_cluster(name=f"projects/{project}/locations/{zone}/clusters/{cluster}")

    # write cluster CA cert
    with NamedTemporaryFile(delete=False) as ca_cert:
        ca_cert.write(base64.b64decode(cluster.master_auth.cluster_ca_certificate))

    # set client configuration
    config = client.Configuration()
    config.host = f'https://{cluster.endpoint}:443'
    config.verify_ssl = True
    config.api_key = {"authorization": "Bearer " + credentials.token}
    config.username = credentials._service_account_email
    config.ssl_ca_cert = ca_cert.name

    return config

def get_pod_status(client, workload_name):
    pods = client.CoreV1Api().list_pod_for_all_namespaces(watch=False)
    pod_status = [i.status.phase for i in pods.items if i.metadata.name.startswith(workload_name)]
    return pod_status[0] if len(pod_status) > 0 else None

def get_job_status(client, workload_name):
    jobs = client.BatchV1Api().list_job_for_all_namespaces(watch=False)
    job_status = [i.status for i in jobs.items if i.metadata.name == workload_name]
    return job_status[0] if len(job_status) > 0 else None

def is_workload_running(workload_type, workload_status):
    flag = (
        (workload_type == 'Pod' and workload_status in ['Pending', 'Running']) or
        (workload_type == 'Job' and workload_status.active)
    )
    return flag
    
def main(argv: Sequence[str]) -> None:
    print(argv)
    global FLAGS 
    FLAGS = flags.FLAGS
    parse_flags(argv)
    flags.FLAGS(argv)
    for attr, flag_obj in flags.FLAGS.__flags.items():
          print(f'{attr}:{flag_obj.value}')
    del argv

    # set log explorer endpoint
    log_uri = "https://console.cloud.google.com/logs/query"
    status = 'Unknown'
    workload_status = None
    pod_status = None
    
    # get credentials and set config
    config = get_config(project=FLAGS.project, zone=FLAGS.zone, cluster=FLAGS.cluster)
    client.Configuration.set_default(config)

    # submit the job
    k8s_api_client = client.ApiClient()
    print(f"Submitting job to the cluster {FLAGS.cluster}")
    response = utils.create_from_yaml(k8s_api_client, FLAGS.job_spec_path, verbose=True)
    # print(response)
    workload_name = response[0][0].metadata.name
    workload_type = response[0][0].kind

    # wait until complete
    pending_time_out = 15*60
    polling_time = 30
    elapsed_time = 0

    # get status
    time.sleep(10)
    
    if workload_type == 'Pod':
        workload_status = get_pod_status(client, workload_name) 
        print(f"Submitted job on cluster {FLAGS.cluster} with pod name={workload_name} with status={workload_status}")
    else:
        workload_status = get_job_status(client, workload_name)
        print(f"Submitted job on cluster {FLAGS.cluster} with job name={workload_name} with {workload_status.active} active jobs")
        
    while is_workload_running(workload_type, workload_status):
        time.sleep(polling_time)
        elapsed_time += polling_time

        if workload_type == 'Pod':
            workload_status = get_pod_status(client, workload_name)
            pod_status = workload_status
        else:
            workload_status = get_job_status(client, workload_name)
            pod_status = get_pod_status(client, workload_name)

        # time out when unable to schedule job
        if pod_status == 'Pending' and elapsed_time > pending_time_out:
            msg = f"Timed out [{pending_time_out}s]. Failed to schedule the job on the cluster {FLAGS.cluster}."
            print(msg)
            print(f"Deleting scheduled pod {workload_name} from {FLAGS.cluster}")
            
            if workload_type == 'Pod':
                client.CoreV1Api().delete_namespaced_pod(name=workload_name,
                                                         namespace="default", 
                                                         body=client.V1DeleteOptions())
            else:
                client.BatchV1Api().delete_namespaced_job(name=workload_name,
                                                          namespace="default", 
                                                          body=client.V1DeleteOptions())
            cluster_log_query = f''';query=
            resource.type="k8s_cluster"
            resource.labels.project_id="{FLAGS.project}"
            resource.labels.cluster_name="{FLAGS.cluster}"
            "{workload_name}"
            severity>=DEFAULT;timeRange=P7D'''
            print(f'Check cluster logs: {log_uri}{urllib.parse.quote(cluster_log_query, safe="")}')            
            raise Exception(msg)
        print(f"Waiting for the job to complete [{polling_time}s]... [status=active]")

    if workload_type == 'Job' and not workload_status.active:
        status = [c.type for c in workload_status.conditions if c.status == 'True'][0]
        print(f"Status of the job {workload_name} = {status}")
    else:
        status = workload_status
        print(f"Status of the pod {workload_name} = {status}")

    # print link to logs
    container_log_query = f''';query=
    resource.type="k8s_container"
    resource.labels.project_id="{FLAGS.project}"
    resource.labels.cluster_name="{FLAGS.cluster}"
    resource.labels.namespace_name="default"
    {'resource.labels.pod_name' if workload_type == 'Pod' else 'labels.k8s-pod/job-name' }="{workload_name}"
    severity>=DEFAULT;timeRange=P7D'''
    print(f'Check container logs for the job: {log_uri}{urllib.parse.quote(container_log_query, safe="")}')

    if status in ['Failed', 'Unknown']:
        raise Exception(f"Failed to submit the job to GKE and run on TPU VM")
    
    return status
    
if __name__ == '__main__':
  app.run(main)