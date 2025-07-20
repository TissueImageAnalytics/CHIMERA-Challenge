#!/bin/bash
echo input folders:
ls  /input/
ls  /input/images/
echo model folder:
ls /opt/ml/model

echo running task_1_inference.py
python -u task_1_inference.py
echo finished