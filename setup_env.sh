#!/bin/bash

python3 -m pip install --upgrade pip && \
   python3 -m pip install virtualenv && \
   python3 -m venv .venv && \
   source .venv/bin/activate && \
   pip install poetry && \
   poetry env use $(which python) && \
   poetry install --no-root
