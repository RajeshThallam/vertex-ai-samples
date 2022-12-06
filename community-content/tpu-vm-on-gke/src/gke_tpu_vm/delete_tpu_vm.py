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

"""Delete a TPU VM

Example command:

python delete_tpu_vm_node.py \
  --tpu_name=tpu-name \
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
    flags.DEFINE_string('zone', None, 'GCP zone of TPU and GKE cluster.')
    flags.DEFINE_string('project', None, 'GCP project of TPU and GKE cluster.')
    flags.DEFINE_string('tpu_name', None, 'TPU VM name.')
    flags.mark_flags_as_required(['tpu_name', 'zone', 'project'])

def delete_tpu_vm(tpu_name: str, zone: str, project: str):
  """Deleting TPU VM"""
  tpu_delete_url = os.path.join('https://tpu.googleapis.com/v2alpha1/projects',
                                project, 'locations', zone, 'nodes', tpu_name)
  print(tpu_delete_url)
  print('Deleting TPU:', tpu_name)

  try:
      creds, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
      creds.refresh(google.auth.transport.requests.Request())
      resp = requests.delete(
          tpu_delete_url,
          headers={'Authorization': f'Bearer {creds.token}'})
      resp_json = resp.json()
      print('Delete TPU response:', resp_json)
      if 'error' in resp_json:
        raise Exception(resp_json["error"]["message"])
      resp.raise_for_status()

      delete_op_url = os.path.join('https://tpu.googleapis.com/v2alpha1',
                                   resp.json()['name'])
      while not resp.json()['done']:
        print('Delete TPU operation still running...')
        time.sleep(30)
        resp = requests.get(
            delete_op_url, headers={'Authorization': f'Bearer {creds.token}'})

      resp_json = resp.json()
      print('Delete TPU operation:', resp_json)
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
          print(f'{attr}: {flag_obj.value}')

  del argv

  try:
    delete_tpu_vm(FLAGS.tpu_name, FLAGS.zone, FLAGS.project)
    print(f'Deleted TPU {FLAGS.tpu_name}')
  except Exception as e:
    print(f"Failed to delete TPU {FLAGS.tpu_name}")
    raise e


if __name__ == '__main__':
  app.run(main)
