#!/bin/bash

source src/venv/bin/activate
src/venv/bin/python3.12 src/CUDA/HashCracking_CUDA.py
deactivate