#!/bin/bash -xv
echo "Copying data from "$1" to /data"
mkdir -p data
gcloud storage cp --recursive $1 ~/data/
