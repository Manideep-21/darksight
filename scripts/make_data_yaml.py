#!/usr/bin/env python
"""Re-write data.yaml with this machine's absolute path (after copying a dataset to Colab/another laptop).

    python scripts/make_data_yaml.py data/exdark_yolo/raw data/exdark_yolo/enhanced
"""

import _bootstrap  # noqa: F401

import sys

from darksight.data.yolo_dataset import write_data_yaml

for root in sys.argv[1:] or ["data/exdark_yolo/raw"]:
    print("wrote", write_data_yaml(root))
