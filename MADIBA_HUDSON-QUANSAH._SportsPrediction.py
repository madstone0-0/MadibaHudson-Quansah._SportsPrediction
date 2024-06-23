#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

get_ipython().system('{sys.executable} -m pip install xgboost dill')


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
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from datetime import datetime


# In[3]:


trainFifa = pd.read_csv("./datasets/male_players (legacy).csv", low_memory=False)
ogFifa = trainFifa.copy()
fifa22 = pd.read_csv("./datasets/players_22.csv", low_memory=False)
pd.set_option("display.max_columns", None)


# # Data  Preprocessing

# In[4]:


trainFifa.info(max_cols=500)


# In[5]:


trainFifa.head()


# In[6]:


fifa22.info(max_cols=500)


# In[7]:


fifa22.head()


# ### Feature Engineering

# In[8]:


# Make composite features for increased correlation
trainFifa["experience"] = datetime.now().year - pd.to_datetime(trainFifa["club_joined_date"]).dt.year
trainFifa["bmi"] = trainFifa["weight_kg"] / (trainFifa["height_cm"] / 100) ** 2
perfMetrics = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
trainFifa["avg_perf"] = trainFifa[perfMetrics].mean(axis=1)
trainFifa.head()


# In[9]:


quantsTrain = trainFifa.select_dtypes(exclude=["object"])
catsTrain = trainFifa[["overall", "work_rate", "preferred_foot", "body_type"]]
catsTrain.head()


# ### Exploratory Data Analysis

# In[10]:


from pandas.plotting import scatter_matrix

corr = quantsTrain.corr()
corr = corr.loc[np.where(corr["overall"].abs() > 0.5, 1, 0) == 1]
potFeats = corr["overall"].sort_values(ascending=False)
top = potFeats
potFeats = [feat for feat in zip(top, top.index)]
potFeats


# In[11]:


# Reset dataset after checking composite feature correlations
trainFifa = trainFifa.drop(["experience", "bmi", "avg_perf"], axis=1)
quantsTrain = trainFifa.select_dtypes(exclude=["object"])
corr = quantsTrain.corr()
potFeats = corr["overall"].abs().sort_values(ascending=False)
top = potFeats[1:11]
potFeats = [feat for feat in zip(top, top.index)]
potFeats


# In[12]:


potFeats = [feat for cor, feat in potFeats]
potFeats


# In[13]:


scatter_matrix(trainFifa[[*potFeats[:5], "overall"]], figsize=(12, 8))


# In[14]:


# Box plot for Categorical to Quantitative Relationship
plt.figure(figsize=(50, 10))
catsTrain.boxplot(column=["overall"], by="preferred_foot")
catsTrain.boxplot(column=["overall"], by="work_rate")
catsTrain.boxplot(column=["overall"], by="body_type")
plt.show()


# In[15]:


from numpy import floor

# Choose highly correlated features and age based on intuition
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

X_train_train.head()


# In[16]:


y_train_train[:5]


# ### Data Cleaning

# In[17]:


def cleanData(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data by imputing missing values, encoding categorical data, and scaling numerical data
    """
    moneyList = ["value_eur", "wage_eur", "release_clause_eur"]

    money = data[moneyList]
    money = money.add_prefix("dim_eur__")
    perfs = data[perfMetrics]
    remainder = data.drop([*moneyList, *perfMetrics], axis=1)
    remainder = remainder.select_dtypes(exclude=object)
    remainder = remainder.add_prefix("remainder__")

    quantImpute = SimpleImputer(strategy="median")
    avgFunc = FunctionTransformer(lambda X: pandas.DataFrame(X.mean(axis=1)), feature_names_out=lambda x, y: ["avg"])
    dimFunc = FunctionTransformer(np.log, feature_names_out="one-to-one")
    scaler = MinMaxScaler()

    money = quantImpute.fit_transform(money)
    money = pd.DataFrame(money, columns=quantImpute.get_feature_names_out())

    perfs = quantImpute.fit_transform(perfs)
    perfs = pd.DataFrame(perfs, columns=quantImpute.get_feature_names_out())

    remainder = quantImpute.fit_transform(remainder)
    remainder = pd.DataFrame(remainder, columns=quantImpute.get_feature_names_out())

    perfs = avgFunc.fit_transform(perfs)
    perfs = pd.DataFrame(perfs, columns=avgFunc.get_feature_names_out())
    perfs = perfs.add_prefix("avg_perf__")

    money = dimFunc.fit_transform(money)
    money = pd.DataFrame(money, columns=dimFunc.get_feature_names_out())

    combined = pd.concat([money, perfs, remainder], axis=1)
    combined = scaler.fit_transform(combined)
    combined = pd.DataFrame(combined, columns=scaler.get_feature_names_out(), index=data.index)
    return combined


# In[18]:


avgPipe = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(lambda X: pandas.DataFrame(X.mean(axis=1)), feature_names_out=lambda x, y: ["avg"]),
)

quantPipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("round", FunctionTransformer(np.round, feature_names_out="one-to-one")),
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
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
)

first = ColumnTransformer(
    [
        ("dim_eur", dimPipe, ["value_eur", "wage_eur", "release_clause_eur"]),
        ("avg_perf", avgPipe, perfMetrics),
        ("cat", catPipe, make_column_selector(dtype_include=[object])),
    ],
    remainder=quantPipe,
)

# Pipeline for cleaning input data on the deployed model
pipe = make_pipeline(first, MinMaxScaler())
pipe


# In[19]:


piped = pd.DataFrame(
    pipe.fit_transform(X_train_train),
    columns=pipe.get_feature_names_out(),
    index=X_train_train.index,
)
piped.head()


# In[20]:


cleanData(X_train_train).head()


# # Training

# In[21]:


def tests(name: str, preds: np.array, actuals: np.array) -> None:
    rmse = root_mean_squared_error(preds, actuals)
    mae = mean_absolute_error(preds, actuals)
    r2s = r2_score(preds, actuals)
    print(f"{name}:\nRMSE: {rmse}\nMAE: {mae}\nR2S: {r2s}\n")


# In[22]:


X_train_train = cleanData(X_train_train)
X_train_test = cleanData(X_train_test)
X_train_train = X_train_train.values
X_train_test = X_train_test.values


# In[23]:


from sklearn.linear_model import LinearRegression

linReg = LinearRegression()
linReg.fit(X_train_train, y_train_train)


# In[24]:


linPreds = linReg.predict(X_train_test)


# In[25]:


from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()
tree.fit(X_train_train, y_train_train)


# In[26]:


treePreds = tree.predict(X_train_test)


# In[27]:


from sklearn.ensemble import RandomForestRegressor
from joblib import parallel_backend

with parallel_backend("threading", n_jobs=-1):
    randSkog = RandomForestRegressor()
    randSkog.fit(X_train_train, y_train_train)
randSkog


# In[28]:


randSkogPreds = randSkog.predict(X_train_test)


# In[29]:


from sklearn.ensemble import GradientBoostingRegressor

with parallel_backend("threading", n_jobs=-1):
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train_train, y_train_train)
gbr


# In[30]:


gbrPreds = gbr.predict(X_train_test)


# In[31]:


from xgboost import XGBRegressor

with parallel_backend("threading", n_jobs=-1):
    xgbr = XGBRegressor()
    xgbr.fit(X_train_train, y_train_train)
xgbr


# In[32]:


xgbrPreds = xgbr.predict(X_train_test)


# # Evaluation

# ## Normal Evaluation

# In[33]:


tests("Multi-Linear", linPreds, y_train_test)
tests("Decision Tree", treePreds, y_train_test)
tests("Random Forest", randSkogPreds, y_train_test)
tests("Gradient Boosting Regressor", gbrPreds, y_train_test)
tests("XGradient Boosting Regressor", xgbrPreds, y_train_test)


# ## Cross Evaluation

# In[34]:


from sklearn.model_selection import cross_val_score


def crossTests(name, model, X, y):
    cross = cross_val_score(
        model, X, y, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    s = pd.Series(cross)
    print(f"{name.upper()}:")
    return s.describe()


# In[35]:


crossTests("Linear Regression", linReg, X_train_test, y_train_test)


# In[36]:


crossTests("Decision Tree Regressor", tree, X_train_test, y_train_test)


# In[37]:


crossTests("Random Forest", randSkog, X_train_test, y_train_test)


# In[38]:


crossTests("Gradient Boosting Regressor", gbr, X_train_test, y_train_test)


# In[39]:


crossTests("XGradient Boosting Regressor", xgbr, X_train_test, y_train_test)


# Looking at the cross evaluation tests it seems the RandomForestRegressor generalizes best to the dataset as it's mean negative root mean squared score is the lowest

# In[40]:


paramDist = {
    "n_estimators": randint(50, 100),
    "max_features": randint(15, 20),
    "max_depth": randint(16, 18),
}

randSearch = RandomizedSearchCV(
    randSkog,
    param_distributions=paramDist,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=42,
    n_jobs=-1,
    n_iter=10,
)
randSearch.fit(X_train_train, y_train_train)


# In[41]:


randSearch.best_params_


# In[42]:


randSearch.best_score_


# In[43]:


finalModel = randSearch.best_estimator_
finalModel


# In[44]:


finalModel = Pipeline([("preprocessing", pipe), ("randSkog", finalModel)])
finalModel


# Pipeline for the preprocessing of new data points on the deployed model

# In[45]:


featureImportances = finalModel["randSkog"].feature_importances_
sorted(
    zip(
        featureImportances.round(2), finalModel["preprocessing"].get_feature_names_out()
    ),
    reverse=True,
)


# # Testing with test set

# In[46]:


finalPreds = finalModel.predict(X_test)
tests("Final Random Forest", finalPreds, y_test)


# In[47]:


crossTests("Final Random Forest", finalModel, X_test, y_test)


# In[48]:


X_test.head()


# In[49]:


y_test[:5]


# ### Calculate confidence level

# In[50]:


import scipy

class ConfidenceLevel:
    def __init__(self, preds, std, confidenceLevel = 1.96) -> None:
        self.preds = preds
        self.cl = confidenceLevel
        self.std = std
    def ci(self, pred) -> str:
        interval = self.cl * self.std
        confPercentage = scipy.stats.norm.cdf(self.cl) - scipy.stats.norm.cdf(-self.cl)
        return f"{pred} ± {interval[0]:.2f} with {confPercentage:.1%} confidence"


# In[51]:


treePreds = np.array([tree.predict(X_train_train) for tree in finalModel["randSkog"].estimators_])
std = np.std(treePreds, axis=0)
conf = ConfidenceLevel(treePreds, std)
conf.ci(np.round(finalPreds[0]))


# # Export model

# In[59]:


from pathlib import Path

def split(src, dest, wsize):
    src = Path(src)
    dest = Path(dest)
    dest.mkdir(exist_ok=True)

    partNum = 0
    with open(src, "rb") as f:
        while True:
            chunk = f.read(wsize)

            if not chunk:
                break

            partNum += 1
            filename = f"{src.stem}-{partNum}.pkl"
            with open(dest / filename, "wb") as p:
                p.write(chunk)
    src.unlink()


# In[60]:


from dill import dump

dump(finalModel, open("./server/Fifa_Model.pkl", mode="wb"))
dump(conf, open("./server/ci.pkl", mode="wb"))

split("./server/Fifa_Model.pkl", "./server/", int(5E7))

