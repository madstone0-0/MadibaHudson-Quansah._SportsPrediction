#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

# !{sys.executable} -m pip install xgboost
get_ipython().system('{sys.executable} -m pip install dill')


# In[2]:


import numpy as np
import pandas as pd
import pandas
import scipy as sp
from sklearn.pipeline import Pipeline, make_pipeline, FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from numpy import int64, float64
import matplotlib.pyplot as plt
from datetime import datetime


# In[3]:


trainFifa = pd.read_csv("./datasets/male_players (legacy).csv", low_memory=False)
ogFifa = trainFifa.copy()
fifa22 = pd.read_csv("./datasets/players_22.csv", low_memory=False)
pd.set_option("display.max_columns", None)


# In[4]:


trainFifa.info(max_cols=500)


# In[5]:


trainFifa.head()


# In[6]:


fifa22.info(max_cols=500)


# In[7]:


fifa22.head()


# In[8]:


trainFifa["experience"] = datetime.now().year - pd.to_datetime(trainFifa["club_joined_date"]).dt.year
trainFifa["bmi"] = trainFifa["weight_kg"] / (trainFifa["height_cm"] / 100 ) ** 2
perfMetrics = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
trainFifa["avg_perf"] = trainFifa[perfMetrics].mean(axis=1)
trainFifa.head()


# In[9]:


quantsTrain = trainFifa.select_dtypes(exclude=["object"])
catsTrain = trainFifa[["overall", "work_rate", "preferred_foot", "body_type"]]
catsTrain.head()


# In[10]:


from pandas.plotting import scatter_matrix

corr = quantsTrain.corr()
potFeatsIndex = corr.index[abs(corr["overall"]) > 0.3]
potFeats = corr["overall"].sort_values(ascending=False)
top = potFeats
potFeats = [feat for feat in zip(top, top.index)]
potFeats

# scatter_matrix(trainFifa[[*potFeats, "overall"]], figsize=(12, 8))


# In[11]:


trainFifa = ogFifa
quantsTrain = trainFifa.select_dtypes(exclude=["object"])
corr = quantsTrain.corr()
potFeats = corr["overall"].sort_values(ascending=False)
top = potFeats[1:11]
potFeats = [feat for feat in zip(top, top.index)]
potFeats


# In[12]:


potFeats = [feat for cor, feat in potFeats]
potFeats


# In[13]:


plt.figure(figsize=(50, 10))
catsTrain.boxplot(column=["overall"], by="preferred_foot")
catsTrain.boxplot(column=["overall"], by="work_rate")
catsTrain.boxplot(column=["overall"], by="body_type")
plt.show()


# In[14]:


from numpy import floor

# chosenFeatures = ["release_clause_eur", "wage_eur", "age", "mentality_composure", "skill_ball_control",
#                   "movement_reactions", "attacking_short_passing", "passing", "dribbling"]

chosenCorFeatures = [*potFeats, "age", *perfMetrics[:-1]]
chosenCorFeatures.remove("potential")
chosenCorFeatures.remove("passing")
chosenCorFeatures.remove("dribbling")
chosenCorFeatures.remove("attacking_short_passing")


# trainFifa.drop(trainFifa[trainFifa["fifa_version"] == 22].index, inplace=True, axis=0)
trainFifa.dropna(thresh=floor(len(trainFifa) * 0.60), inplace=True, axis=1)



y_train = trainFifa[["overall"]].values.ravel()
X_train = trainFifa[chosenCorFeatures]
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(
    X_train, y_train, random_state=42, test_size=0.2
)

y_test = fifa22[["overall"]].values.ravel()
X_test = fifa22[chosenCorFeatures]
# X_test = fifa22.drop("overall", axis=1)
X_train_train.head()


# In[15]:


y_train_train[:5]


# In[16]:


maxIter = 20

def colAvg(X):
    return pd.DataFrame(X.mean(axis=1))


def avgName(functionTrans, featureNamesIn):
    return ["avg"]


avgPipe = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(lambda X: pandas.DataFrame(X.mean(axis=1)), feature_names_out=lambda x, y: ["avg"]),
    )



quantPipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("round",FunctionTransformer(np.round, feature_names_out="one-to-one")),
        # ("impute", IterativeImputer(max_iter=maxIter, random_state=42)),
        # ("scale", StandardScaler())
    ]
)

catPipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ]
)

dimPipe = make_pipeline(
    SimpleImputer(strategy="median"),
    # IterativeImputer(max_iter=maxIter, random_state=42),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    # FunctionTransformer(np.sqrt, feature_names_out="one-to-one"),
)

first = ColumnTransformer(
    [
        ("dim_eur", dimPipe, ["value_eur", "wage_eur", "release_clause_eur"]),
        ("avg_perf", avgPipe, perfMetrics),
        ("cat", catPipe, make_column_selector(dtype_include=[object])),
    ],
    remainder=quantPipe,
)
pipe = make_pipeline(first, MinMaxScaler())
pipe


# In[17]:


piped = pd.DataFrame(
    pipe.fit_transform(X_train_train),
    columns=pipe.get_feature_names_out(),
    index=X_train_train.index,
)
piped


# In[18]:


def tests(name, preds, actuals):
    rmse = root_mean_squared_error(preds, actuals)
    mae = mean_absolute_error(preds, actuals)
    r2s = r2_score(preds, actuals)
    print(f"{name}:\nRMSE: {rmse}\nMAE: {mae}\nR2S: {r2s}\n")


# In[19]:


from sklearn.linear_model import LinearRegression

linPipe = Pipeline([("preprocessing", pipe), ("linear", LinearRegression())])
linPipe.fit(X_train_train, y_train_train)


# In[20]:


linPreds = linPipe.predict(X_train_test)


# In[21]:


from sklearn.tree import DecisionTreeRegressor

treePipe = Pipeline([("preprocessing", pipe), ("tree", DecisionTreeRegressor())])
treePipe.fit(X_train_train, y_train_train)


# In[22]:


treePreds = treePipe.predict(X_train_test)


# In[23]:


from sklearn.ensemble import RandomForestRegressor
from joblib import parallel_backend
from sklearn.feature_selection import SelectFromModel

with parallel_backend("threading", n_jobs=-1):
    randSkogPipe = Pipeline(
        [
            ("preprocessing", pipe),
            ("randSkog", RandomForestRegressor(warm_start=True)),
        ]
    )
    randSkogPipe.fit(X_train_train, y_train_train)
randSkogPipe


# In[24]:


randSkogPreds = randSkogPipe.predict(X_train_test)


# In[25]:


from sklearn.ensemble import GradientBoostingRegressor

with parallel_backend("threading", n_jobs=-1):
    gbrPipe = Pipeline(
        [
            ("preprocessing", pipe),
            ("gbr", GradientBoostingRegressor()),
        ]
    )
    gbrPipe.fit(X_train_train, y_train_train)
gbrPipe


# In[26]:


gbrPreds = gbrPipe.predict(X_train_test)


# In[27]:


# from xgboost import XGBRegressor

# with parallel_backend("threading", n_jobs=-1):
#     xgbrPipe = Pipeline(
#         [
#             ("preprocessing", pipe),
#             ("xgbr", XGBRegressor()),
#         ]
#     )
#     xgbrPipe.fit(X_train_train, y_train_train)
# xgbrPipe


# In[28]:


# xgbrPreds = xgbrPipe.predict(X_train_test)


# In[29]:


tests("Multi-Linear", linPreds, y_train_test)
tests("Decision Tree", treePreds, y_train_test)
tests("Random Forest", randSkogPreds, y_train_test)
tests("Gradient Boosting Regressor", gbrPreds, y_train_test)
# tests("XGradient Boosting Regressor", xgbrPreds, y_train_test)


# In[30]:


from sklearn.model_selection import cross_val_score


def crossTests(name, model, X, y):
    cross = cross_val_score(
        model, X, y, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    s = pd.Series(cross)
    print(f"{name.upper()}:")
    return s.describe()


# In[31]:


crossTests("Random Forest", randSkogPipe, X_train_test, y_train_test)


# In[32]:


crossTests("Gradient Boosting Regressor", gbrPipe, X_train_test, y_train_test)


# In[33]:


# crossTests("XGradient Boosting Regressor", xgbrPipe, X_train_test, y_train_test)


# In[34]:


paramDist = {
    "randSkog__n_estimators": randint(50, 100),
    "randSkog__max_features": randint(15, 20),
    "randSkog__max_depth": randint(16, 18),
}

randSearch = RandomizedSearchCV(
    randSkogPipe,
    param_distributions=paramDist,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=42,
    n_jobs=-1,
    n_iter=10,
)
randSearch.fit(X_train_train, y_train_train)


# In[35]:


randSearch.best_params_


# In[36]:


randSearch.best_score_


# In[37]:


finalModel = randSearch.best_estimator_
finalModel


# In[38]:


featureImportances = finalModel["randSkog"].feature_importances_
sorted(
    zip(
        featureImportances.round(2), finalModel["preprocessing"].get_feature_names_out()
    ),
    reverse=True,
)


# In[39]:


finalPreds = finalModel.predict(X_test)
tests("Final Random Forest", finalPreds, y_test)


# In[40]:


crossTests("Final Random Forest", finalModel, X_test, y_test)


# In[41]:


X_test.tail()


# In[42]:


y_test[-5:]


# In[43]:


from dill import dump

dump(finalModel, open("../server/Fifa_Model.pkl", mode="wb"))


# In[ ]:




