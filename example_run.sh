#!/bin/bash

CHECKPOINT_DIR=~/.llama/checkpoints/Meta-Llama3.1-8B-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun example.py $CHECKPOINT_DIR