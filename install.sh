conda activate mace_al_clean

pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

pip install torch-geometric
