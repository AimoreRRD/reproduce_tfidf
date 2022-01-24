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