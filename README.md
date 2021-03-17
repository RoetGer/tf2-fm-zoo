# tf2-fm-zoo
Python package for the factorization machine implementations from [tensorflow2_model_zoo](https://github.com/ryancheunggit?tab=repositories).

## Acknowledgement
The original implementation for the methods in this repo were done by [Ren Zhang](https://github.com/ryancheunggit/) who kindly granted permission to use his code for the creation of the package.

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

## Supported Models  

| Model | Reference | Year |
|-------|-----------|------|
| [FM](fm/fm.py) | [Factorization Machines](https://ieeexplore.ieee.org/abstract/document/5694074) | 2010 |
| [FFM](fm/ffm.py) | [Field-aware factorization machines for CTR prediction](https://dl.acm.org/citation.cfm?id=2959134) | 2016 |
| [FNN](fm/fnn.py) | [Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/abs/1601.02376) | 2016 |
| [AFM](fm/afm.py) | [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/abs/1708.04617) | 2017 |
| [DeepFM](fm/dfm.py) | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) | 2017 |
| [NFM](fm/nfm.py) | [Nerual Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/abs/1708.05027) | 2017 |
| [xDeepFM](fm/xdfm.py) | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170) | 2018 |
| [AutoInt](fm/afi.py) | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) | 2018 |
| [FNFM](fm/fnfm.py) | [Field-aware Neural Factorization Machine for Click-Through Rate Prediction](https://arxiv.org/abs/1902.09096) | 2019 |
