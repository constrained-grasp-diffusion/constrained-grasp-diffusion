#!/bin/bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118  torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+118.html
conda install conda-forge::suitesparse
conda install -c conda-forge scikit-sparse
pip install theseus-ai==0.1.3
pip install -r requirements.txt
pip install -e .
pip install pykdtree
pip install open3d
