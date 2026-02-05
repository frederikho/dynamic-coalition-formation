#!/bin/bash
# Wrapper script to build static data with venv activated

# Activate venv if it exists, otherwise use system python
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python3 build_static_data.py
