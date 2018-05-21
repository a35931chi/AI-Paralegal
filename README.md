## Introduction
last update: 2018/5/21


This is an exercise to try to help lawyers deal with case updates. Our algo needs to accomplish the following:
* Correctly classify a new case update
* Select the appropriate response to that case update

Analysis done in python IDLE 3.5.3. 
Please refer to the [blank](blank) and [blank](blank) for the complete project and report details.


## Please refer to:
* [Project Summary](): Detailing project motivation
* [NLP, Train Model & Visualize](): Detailing cleanup, keywords identification, and topic modeling processes
* [Test Model](): Detailing testing methodologies

## References:
### Utility Libraries:
```
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import visuals as vs #from one of the MLND project. Was useful in visualizing PCA
from time import gmtime, strftime, time
import warnings
```

### NLP Libraries:
```
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import norm, skew 
```
### Topic Modeling Libraries:
```
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
```

