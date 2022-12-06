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

"""Create a TPU VM to use in GKE Cluster.

This script is highly experimental. TPU VM GKE nodes may not behave as expected.

Example command:

python create_tpu_vm_node.py \
  --cluster=$USER-cluster \
  --zone=us-central1-a \
  --project=$USER-gke-dev
"""

import os
import sys
import time
from typing import Dict, Sequence

from absl import app
from absl import flags
import google.auth
from google.cloud import compute_v1
from google.cloud import container_v1
import requests

def parse_flags(args):
    flags.DEFINE_string('cluster', None, 'Name of GKE cluster')
    flags.DEFINE_string('zone', None, 'GCP zone of TPU and GKE cluster.')
    flags.DEFINE_string('project', None, 'GCP project of TPU and GKE cluster.')

    flags.DEFINE_string('tpu_name', 'gke-tpu', 'TPU VM name.')
    flags.DEFINE_string('accelerator_type', 'v3-8', 'TPU VM type.')
    flags.DEFINE_string('runtime_version', 'v2-alpha-cos',
                        'TPU VM runtime version')
    flags.DEFINE_bool('create_tpu_pool', True,
                      'Whether to create TPU node pool if not present.')
    flags.DEFINE_string(
        'tpu_pool', 'tpu-pool',
        'Name of node pool in `cluster` to create TPU VM Nodes in.')
    flags.DEFINE_string('tpu_pool_image_type', 'cos',
                        'Type of image in TPU pool (e.g. "cos", "cos_containerd")')
    flags.DEFINE_string('alias_ip_netmask', None,
                        'Netmask to use for Alias IP Range of TPU VMs (e.g. "/28"). '
                        'use_alias_ip flag is required to be set to use this feature')
    flags.DEFINE_string('alias_ip_range_name', None,
                        'Subnet range name from which Alias IP range for TPU VMs '
                        'will be allocated. '
                        'If not set, Alias IP range will be allocated from primary range.'
                        'use_alias_ip flag is required to be set to use this feature')
    flags.DEFINE_bool(
      'tpu_node_labels', True,
      'Whether to add tpu.googleapis.com/{node,type} labels to the GKE node.')
    flags.DEFINE_bool(
        'use_alias_ip', False,
        'Whether to create TPU VMs on VPC native cluster (using Alias IP).')

    flags.register_validator(
          'alias_ip_netmask',
          lambda x: False if x and not FLAGS.use_alias_ip else True,
          'Must set use_alias_ip flag to use Alias IPs.')
    flags.register_validator(
          'alias_ip_range_name',
          lambda x: False if x and not FLAGS.use_alias_ip else True,
          'Must set use_alias_ip flag to use Alias IPs.')
    flags.mark_flags_as_required(['cluster', 'zone', 'project'])

def create_tpu_pool(cluster_id: str, tpu_pool: str, image_type: str, zone: str,
                    project: str):
  """Create empty pool for TPU VM nodes if not present."""
  print(f'cluster_id: {cluster_id}')
  print(f'tpu_pool: {tpu_pool}')
  print(f'image_type: {image_type}')
  print(f'zone: {zone}')
  print(f'project: {project}')
  clusters_client = container_v1.ClusterManagerClient()
  try:
    clusters_client.get_node_pool(
        project_id=project,
        zone=zone,
        cluster_id=cluster_id,
        node_pool_id=tpu_pool)
    print('TPU pool already exists:', tpu_pool)
    return
  except google.api_core.exceptions.NotFound:
    pass

  print('Creating TPU pool:', tpu_pool)

  node_pool = container_v1.NodePool(
      name=tpu_pool,
      initial_node_count=0,
      config=container_v1.NodeConfig(
          image_type=image_type),
      management=container_v1.NodeManagement(
          auto_repair=False,
          auto_upgrade=False,
      ),
  )
  op = clusters_client.create_node_pool(
      project_id=project, zone=zone, cluster_id=cluster_id, node_pool=node_pool)

  while op.status in (container_v1.Operation.Status.RUNNING,
                      container_v1.Operation.Status.PENDING):
    print('Create pool operation status:', op.status)
    time.sleep(30)
    op = clusters_client.get_operation(
        project_id=project, zone=zone, operation_id=op.name)

  print('Create node pool op:', op)


def get_instance_template(
    cluster_id: str, tpu_pool: str, zone: str,
    project: str) -> compute_v1.types.compute.InstanceTemplate:
  """Get the instance group ID for the TPU Node Pool."""
  clusters_client = container_v1.ClusterManagerClient()
  cluster = clusters_client.get_node_pool(
      project_id=project,
      zone=zone,
      cluster_id=cluster_id,
      node_pool_id=tpu_pool)
  group_id = os.path.basename(cluster.instance_group_urls[0])
  print('Instance group URL:', cluster.instance_group_urls[0])

  groups_client = compute_v1.InstanceGroupManagersClient()
  group = groups_client.get(
      project=project, zone=zone, instance_group_manager=group_id)
  template_id = os.path.basename(group.instance_template)
  print('Instance template URL:', group.instance_template)

  templates_client = compute_v1.InstanceTemplatesClient()
  template = templates_client.get(
      project=project, instance_template=template_id)

  return template


def create_tpu_vm(template: compute_v1.types.compute.InstanceTemplate,
                  tpu_name: str, accelerator_type: str, runtime_version: str,
                  zone: str, project: str, node_labels: Dict[str, str],
                  use_alias_ip: bool, netmask: str, range_name: str):
  """Create the TPU instance from instance template."""
  # Use `requests` because TPU client library doesn't have canIpForward yet.
  tpu_create_url = os.path.join('https://tpu.googleapis.com/v2alpha1/projects',
                                project, 'locations', zone, 'nodes')
  params = {'nodeId': tpu_name}

  metadata = {row.key: row.value for row in template.properties.metadata.items}
  metadata['kube-labels'] = ','.join((metadata['kube-labels'], *(f'{k}={v}' for k, v in node_labels.items())))

  request = {
      'acceleratorType': accelerator_type,
      'runtimeVersion': runtime_version,
      'metadata': metadata,
      'networkConfig': {
          'enableExternalIps': True,
          'canIpForward': not use_alias_ip,
      },
      'tags': list(template.properties.tags.items)
  }
  if use_alias_ip:
    request['networkConfig']['aliasIpRanges'] = [{
        'ipCidrNetmask': netmask,
        'subnetworkRangeName': range_name,
    },]

  print('Creating TPU:', tpu_name)

  try:
      creds, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
      creds.refresh(google.auth.transport.requests.Request())
      resp = requests.post(
          tpu_create_url,
          params=params,
          json=request,
          headers={'Authorization': f'Bearer {creds.token}'})
      resp.raise_for_status()

      create_op_url = os.path.join('https://tpu.googleapis.com/v2alpha1',
                                   resp.json()['name'])
      while not resp.json()['done']:
        print('Create TPU operation still running...')
        time.sleep(30)
        resp = requests.get(
            create_op_url, headers={'Authorization': f'Bearer {creds.token}'})

      resp_json = resp.json()
      print('Create TPU operation:', resp_json)
      if 'error' in resp_json:
        raise Exception(resp_json["error"]["message"])
  except Exception as e:
    raise e


def main(argv: Sequence[str]) -> None:
  print(argv)
  global FLAGS 
  FLAGS = flags.FLAGS
  parse_flags(argv)
  flags.FLAGS(argv)
  for attr, flag_obj in flags.FLAGS.__flags.items():
          print(f'{attr}:{flag_obj.value}')

  del argv

  if FLAGS.create_tpu_pool:
    create_tpu_pool(FLAGS.cluster, FLAGS.tpu_pool, FLAGS.tpu_pool_image_type,
                    FLAGS.zone, FLAGS.project)
  template = get_instance_template(FLAGS.cluster, FLAGS.tpu_pool, FLAGS.zone,
                                   FLAGS.project)

  tpu_node_labels = {
    'tpu.googleapis.com/name': FLAGS.tpu_name,
    'tpu.googleapis.com/type': FLAGS.accelerator_type,
  } if FLAGS.tpu_node_labels else {}
    
  try:  
    create_tpu_vm(template, FLAGS.tpu_name, FLAGS.accelerator_type,
                  FLAGS.runtime_version, FLAGS.zone, FLAGS.project,
                  tpu_node_labels, FLAGS.use_alias_ip, FLAGS.alias_ip_netmask,
                  FLAGS.alias_ip_range_name)
    print(f'Created TPU {FLAGS.tpu_name} for cluster {FLAGS.cluster}')
  except Exception as e:
    print(f"Failed to create TPU {FLAGS.tpu_name}")
    raise e


if __name__ == '__main__':
  app.run(main)
