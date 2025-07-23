#!/bin/bash
echo input folders:
ls  /input/
ls  /input/images/
echo model folder:
ls /opt/ml/model

echo app folder:
ls /opt/app

echo running task_1_inference_template.py
python -u task1_inference_template.py
echo finished