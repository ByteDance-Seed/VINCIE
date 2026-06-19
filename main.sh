#!/bin/bash

echo "Launch run script"
torchrun --nproc-per-node=${NGPU:-1} main.py "$@"

