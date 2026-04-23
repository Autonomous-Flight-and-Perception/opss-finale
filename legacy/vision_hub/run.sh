#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
test -d venv39 || python3.9 -m venv venv39
source venv39/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
export USE_TPU="${USE_TPU:-1}"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
