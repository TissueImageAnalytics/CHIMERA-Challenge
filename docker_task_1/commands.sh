#!/bin/bash
echo input folders:
ls  /input/images/kidney-transplant-biopsy-wsi-pas/
ls /input/images/tissue-mask/
echo model folder:
ls /opt/ml/model

echo running task_1_inference.py
python -u task_1_inference.py
echo finished