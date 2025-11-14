#!/bin/bash
export DYLD_LIBRARY_PATH=/opt/anaconda3/envs/myenv/lib:$DYLD_LIBRARY_PATH
cd "$(dirname "$0")"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

