# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: reproduce_tfidf
#     language: python
#     name: reproduce_tfidf
# ---

# %% tags=[]
import pickle as pkl
import numpy as np
from pathlib import Path

# %% [markdown]
# # Load Features 

# %%
root_path = Path('').cwd()

features_1_path = root_path / 'data/data_transformed_dense-AD-C02D3RFQMD6M.pkl'
features_2_path = root_path / 'data/data_transformed_dense-ip-172-31-10-252.pkl'

# %% tags=[]
with open(features_1_path, 'rb') as fl:
    features_1 = pkl.load(fl)

with open(features_2_path, 'rb') as fl:
    features_2 = pkl.load(fl)

# %% tags=[]
print(features_1.shape)
features_1

# %% tags=[]
print(features_2.shape)
features_2

# %% [markdown]
# # Compare Features 

# %% tags=[]
diff = np.array(features_1) - np.array(features_2)
print(f'Difference between features: \n{diff}')

# %% tags=[]
print(f'Total size: {diff.size}')
print(f'Non zero elements: {np.count_nonzero(diff)}')

# %% tags=[]
print(f"Max difference: {np.max(diff)}")
print(f"Min difference: {np.min(diff)}")

# %% tags=[]
equal = np.array(features_1) == np.array(features_2)
assert np.all(equal), 'Not all values are the same'
