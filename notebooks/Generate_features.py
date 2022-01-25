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
import pandas as pd
from microcosm.config.validation import typed
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import pickle as pkl
import socket
from pathlib import Path

# %% [markdown]
# # Load Data

# %% pycharm={"name": "#%%\n"} tags=[]
root_path = Path('').cwd()
single_data_path = root_path / '../data/single_client_sample_data.pkl'
single_client_data = pd.read_pickle(single_data_path)
print(f"Data size: {len(single_client_data)}")
single_client_data.head()


# %% [markdown]
# # Featurizer

# %% tags=[]
class Featurizer:      
    config = {
    'url_vect':{
        "input_col":  "url",
        "augment":  False,
        "data":  "train",
        "featurizer":  "tfidf",
        "sublinear_tf":  False,
        "tfidf_lowercase":  False,
        "tfidf_max_df":  0.8,
        "tfidf_max_features":  50_000,
        "tfidf_min_df":  1,
        "tfidf_ngram_range_max":  2,
        "tfidf_ngram_range_min":  1,
    },
    'html_vect':{
        "input_col": "html_simplified",
        "augment": False,
        "data": "train",
        "featurizer": "tfidf",
        "sublinear_tf": True,
        "tfidf_lowercase": False,
        "tfidf_max_df": 0.9,
        "tfidf_max_features": 90_000,
        "tfidf_min_df": 0.2,
        "tfidf_ngram_range_max": 2,
        "tfidf_ngram_range_min": 1,
    },
    }
    def __init__(self):        
        self.col_features = [
            "url_simplified",
            "html_simplified",
            "html_simplified_len",
            "url_simplified_len",
            "url_simplified_segmentations_count",
            "html_simplified_segmentations_count",
        ]
    
        self.url_vectorizer = self.create_vectorizers("url")
        self.html_vectorizer = self.create_vectorizers("html")
        self.transformer = self.create_tranformer()


    def create_vectorizers(self, name):
        tokenizer = None

        return TfidfVectorizer(
            analyzer="word",
            lowercase=False,
            max_df=self.config[f"{name}_vect"]["tfidf_max_df"],
            max_features=self.config[f"{name}_vect"]["tfidf_max_features"],
            min_df=self.config[f"{name}_vect"]["tfidf_min_df"],
            ngram_range=(
                self.config[f"{name}_vect"]["tfidf_ngram_range_min"],
                self.config[f"{name}_vect"]["tfidf_ngram_range_max"],
            ),
            tokenizer=tokenizer,
        )

    def create_tranformer(self):
        return ColumnTransformer(
            [
                ("url_tfidf", self.url_vectorizer, "url_simplified"),
                ("html_tfidf", self.html_vectorizer, "html_simplified"),
                (
                    "scaler",
                    MinMaxScaler(),
                    [
                        "html_simplified_len",
                        "url_simplified_len",
                        "url_simplified_segmentations_count",
                        "html_simplified_segmentations_count",
                    ],
                ),
            ],
            remainder="drop",
        )

    def fit(self, input_data: pd.DataFrame):        
        df_train = input_data[self.col_features]
        self.transformer.fit(df_train)        

    def transform(self, input_data: pd.DataFrame) -> csr_matrix:        
        return self.transformer.transform(input_data[self.col_features])    

# %% tags=[]
f = Featurizer()

# %% [markdown]
# ## Train featurizer

# %% tags=[]
f.fit(single_client_data)

# %% tags=[]
data_transformed = f.transform(single_client_data)

# %% tags=[]
data_transformed_dense = data_transformed.todense()

# %% tags=[]
data_transformed_dense

# %% [markdown]
# ## Save transformed data for comparison

# %% tags=[]
host_name = socket.gethostname()

print(host_name)

save_path = root_path / f'../data/data_transformed_dense-{host_name}.pkl'

print(f'Saved data_transformed_dense at: {save_path}')
with open(save_path, 'wb') as fs:
    pkl.dump(data_transformed_dense, fs)

# %%
