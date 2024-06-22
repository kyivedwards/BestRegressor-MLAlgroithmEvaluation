
#######################
# COMP432 - G01
# Part 2 - Regressors
#######################


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import sklearn.gaussian_process as gp
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer


######## SETUP ########

TRAIN_SIZE = 0.8
MODEL_TYPES = ['Linear', 'SVM', 'DecisionTree', 'RandomForest', 'KNN', 'AdaBoost', 'GaussianProcess', 'NeuralNetwork']
DATA_SOURCES = ['wine', 'communities', 'qsar', 'facebook', 'bike', 'student', 'concrete', 'sgemm']

# Store data sets in order of DATA_SOURCES.
X_training_sets = []
X_testing_sets = []
y_training_sets = []
y_testing_sets = []


def store_data(X, y):
    imp = SimpleImputer(strategy='mean')
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), train_size=TRAIN_SIZE, random_state=0)
    X_training_sets.append(imp.fit_transform(X_train))
    X_testing_sets.append(imp.fit_transform(X_test))
    y_training_sets.append(y_train)
    y_testing_sets.append(y_test)


######## LOAD DATE FILES ########

## Dataset 1: Wine Quality
# Import data
wine_df = pd.read_csv('datasets/winequality-red.csv', sep=';', dtype=float)
wine_df
# wine_df

# Configure & store data
X = wine_df.drop('quality', axis=1)
Y = wine_df['quality']
store_data(X, Y)

## 2. Communities and Crime
names = list()
with open("datasets/communities.names", "r") as f:
    for line in f.readlines():
        line = line.split(" ")
        if line[0] == "@attribute":
            names.append(line[1])
comm_dataset = pd.read_csv("datasets/communities.data", delimiter=',', names=names)

# feature selection
L_drop = [comm_dataset.columns[0], comm_dataset.columns[1], comm_dataset.columns[2], comm_dataset.columns[3]]
for i in range(11, len(comm_dataset.columns)):
    L_drop.append(comm_dataset.columns[i])
comm_df = comm_dataset.drop(columns=L_drop)

comm_df['y'] = comm_dataset['state']
comm_df = comm_df.astype(float)

X = comm_df.drop('y', axis=1)
Y = comm_df['y']
store_data(X, Y)

## 3. QSAR aquatic toxicity
# Importing data
toxic_df = pd.read_csv('datasets/qsar_aquatic_toxicity.csv', sep=';', names=["TPSA", "SAacc", "H-050", "MLOGP", "RDCHI", "GATS1p", "nN", "C-040", "LC50"], dtype=float)
# toxic_df

X = toxic_df.drop('LC50', axis=1)
Y = toxic_df['LC50']
store_data(X, Y)

## 4. Facebook metrics
# Importing data
fb_df = pd.read_csv('datasets/dataset_Facebook.csv', sep=';', header=0)
features_cat = ["Type"]

# Replace the columns with categorical values by its one hot encoding
for feat in features_cat:
    tmp_df = pd.get_dummies(fb_df[feat], prefix=feat)
    idx = fb_df.columns.get_loc(feat)
    for i, col in enumerate(tmp_df.columns):
        fb_df.insert(idx + i, col, tmp_df[col])
    fb_df = fb_df.drop(feat, axis=1)
# fb_df
# Delete every row containing at least one 'Nan'
fb_df = fb_df.dropna(axis=0).astype(int)

X = fb_df.drop("Total Interactions", axis=1)
Y = fb_df["Total Interactions"]
store_data(X, Y)

## 5. Bike Sharing (use hour data)
# Importing data
bike = pd.read_csv('datasets/hour.csv')
bike = bike.drop(['instant', 'dteday'], axis=1)
# bike

X = bike.drop('cnt', axis=1)
Y = bike['cnt']
store_data(X, Y)


## 6. Student Performance
# Importing data
stu = pd.read_csv('datasets/student-por.csv', sep=';')
features_cat = ['school', "sex", 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

# replace the columns with categorical values by its one hot encoding
for feat in features_cat:
    tmp_df = pd.get_dummies(stu[feat], prefix=feat)
    idx = stu.columns.get_loc(feat)
    for i, col in enumerate(tmp_df.columns):
        stu.insert(idx + i, col, tmp_df[col])
    stu = stu.drop(feat, axis=1)
# stu

X = stu.drop("G2", axis=1)
Y = stu["G2"]
store_data(X, Y)

## 7. Concrete Compressive Strength
# Importing data
con = pd.read_excel('datasets/Concrete_Data.xls')
# con

X = con.drop(['Concrete compressive strength(MPa, megapascals) '], axis=1)
Y = con['Concrete compressive strength(MPa, megapascals) ']
store_data(X, Y)

## 8. SGEMM GPU kernel performance 
sge = pd.read_csv('datasets/sgemm_product.csv')
sge = sge.iloc[-10000:]

X = sge.drop('Run4 (ms)', axis=1)
Y = sge['Run4 (ms)']
store_data(X, Y)


# Models used for testing. Re-initialized for each dataset
def generate_models():
    # Generate and store default models
    classification_models = []
    # Linear Regression
    classification_models.append(linear_model.LinearRegression())
    # SVM
    classification_models.append(SVR(kernel="rbf", C=100, max_iter=10000, gamma=0.1, epsilon=0.1))
    # Decision Tree
    classification_models.append(DecisionTreeRegressor(max_depth=100, random_state=0))
    # Random Forest
    classification_models.append(RandomForestRegressor(random_state=0))
    # K-nearest Neighbors
    classification_models.append(KNeighborsRegressor(n_neighbors=8))
    # AdaBoost
    classification_models.append(AdaBoostRegressor(random_state=0, n_estimators=100))
    # Gaussian Process
    classification_models.append(gp.GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1, normalize_y=True, random_state=0))
    # Neural Network
    classification_models.append(MLPRegressor(random_state=0))

    return classification_models


######## TRAIN MODELS ########

trained_models = []
for index, source in enumerate(DATA_SOURCES):
    print("Training: " + source + " with index " + str(index))
    trained_models.append([model.fit(X_training_sets[index], y_training_sets[index]) for model in generate_models()])


######## TEST MODELS ########

results = np.zeros(64)
for index_src, source in enumerate(DATA_SOURCES):
    print("Testing: " + source)
    for index, model in enumerate(trained_models[index_src]):
        results[index_src * 8 + index] = r2_score(y_testing_sets[index_src], model.predict(X_testing_sets[index_src]))
print("Done")


######## DISPLAY RESULTS ########

results = results.reshape(8, 8)
print('\t', end='\t')
print(*MODEL_TYPES, sep='\t')
for index, row in enumerate(results):
    print(DATA_SOURCES[index], end='\t')
    print(*row, sep='\t')
