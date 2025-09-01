#!/usr/bin/env bash
set -euo pipefail   # -e: exit on error, -u: unset var is error, pipefail: catch pipe errors
# optional: nicer error info
trap 'echo "Error on line $LINENO (exit $?)"; exit 1' ERR

echo input folders:
ls  /input/
ls  /input/images/
echo model folder:
ls /opt/ml/model

echo app folder:
ls /opt/app



cd /opt/app

echo running clinical_only_inference.py
exec python clinical_only_inference.py
echo finished

