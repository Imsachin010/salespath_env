#!/usr/bin/env bash
# Simple alias for the main entrypoint
cd /app
exec ./run_training.sh "$@"
