# %%
import numpy as np
import polars as pl
from sklearn.decomposition import PCA


def pca_cutoffs(data: pl.DataFrame, pct: float=95):
    threshold = np.percentile(data['delay'], pct)
    mask = data['delay'] >= threshold

    x_high = data.filter(mask).select(pl.exclude('delay')).to_numpy()

    pca = PCA(n_components=1)
    pca.fit(x_high)
    cutoff = pca.components_[0]
    return cutoff, threshold

