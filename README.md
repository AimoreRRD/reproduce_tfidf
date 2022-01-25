# reproduce_tfidf
This repo tries to reproduce scikit-learn tfidf in different machines.

### [Machine 1](https://wandb.ai/globality/web-annotation-extractor/runs/36yncqdv/overview) (AKA aimore_machine)
### [Machine 2](https://wandb.ai/globality/web-annotation-extractor/runs/2weidsas/overview) (AKA marc_machine)

# Installation
```
yes | conda create -n reproduce_tfidf python=3.8;
conda activate reproduce_tfidf;
pip install pip-tools;
pip-sync requirements.txt;
```
# Data
To download the data for the featurizer:
```
cd data/
dvc pull
```

# Run
There are two python codes in notebooks:
1. `Generate_features.py`: Based on the downloaded data with dvc creates the features and saves it at `data/features/`
```
python notebooks/Generate_features.py
```

2. `Compare_features.py`: Compare two features. Make sure to point to the features (change the path inside the code) 
```
python notebooks/Compare_features.py
```