#!/bin/bash

set -e
set -u

echo "Running EDMD analysis..."
uv run python scripts/perform_edmd.py
echo "Done."
