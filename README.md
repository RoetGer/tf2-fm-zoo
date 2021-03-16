# tf2-fm-zoo
Python package repo for the factorization machine implementations from the [tensorflow2_model_zoo](https://github.com/ryancheunggit?tab=repositories) repo.

## Installation

```bash
pip install tf2_fm_zoo
```
## Basic Example

```python
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import load_boston

from fm_zoo.fm import FactorizationMachine


X, y = load_boston(return_X_y=True)

X = X[:,:3]
y = tf.cast(y, dtype=tf.float32)

kbd = KBinsDiscretizer(n_bins=15, encode="ordinal")

nunique_vals = pd.DataFrame(X).nunique()
X = tf.cast(kbd.fit_transform(X), dtype=tf.int64)

fm = FactorizationMachine(
    feature_cards=tf.cast(nunique_vals, tf.int32), 
    factor_dim=3)

fm.compile(loss=tf.keras.losses.mean_squared_error, optimizer="Adam")
hist = fm.fit(
    X, y, 
    validation_split=0.15, 
    batch_size=16,
    epochs=100,
    callbacks=[
      tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ])

pd.DataFrame(hist.history).plot(figsize=(15,10))
```

## Acknowledgement
The original implementation for the methods in this repo were done by [Ren Zhang](https://github.com/ryancheunggit/) who kindly granted permission to use his code for the creation of the package.

