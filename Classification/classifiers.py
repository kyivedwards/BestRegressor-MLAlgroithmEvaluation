
#######################
# COMP432 - G01
# Part 1 - Classifiers
#######################


import pandas as pd
import numpy as np

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.impute import SimpleImputer

from scipy.io.arff import loadarff


######## SETUP ########

TRAIN_SIZE = 0.8
MODEL_TYPES = ["Linear Regression", "SVM", "Decision Tree", "Random Forest", "K-nearest Neighbors", "AdaBoost", "Gaussian Naive Bayes", "Neural Network"]
DATA_SOURCES = ["Diabetic Retinopathy", "Default of Credit Card Clients", "Breast Cancer Wisconsin", "Statlog", "Adult", "Yeast", "Thoracic Surgery", "Seismic_Bumps"]

# Store data sets in order of DATA_SOURCES.
X_training_sets = []
X_testing_sets = []
y_training_sets = []
y_testing_sets = []


def store_data(X, y):
    imp = SimpleImputer(strategy='mean')
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(np.array(X), np.array(y), train_size=TRAIN_SIZE, random_state=0)
    X_training_sets.append(imp.fit_transform(X_train))
    X_testing_sets.append(imp.fit_transform(X_test))
    y_training_sets.append(y_train)
    y_testing_sets.append(y_test)


######## LOAD DATE FILES ########

## 1. Diabetic Retinopathy
# Import data
raw_data = loadarff('datasets/diabetic_messidor_features.arff')
diabetes_df = pd.DataFrame(raw_data[0])
diabetes_df.rename(columns={'0': 'Quality',
                            '1': 'Pre Screening Result',
                            '2': '#MAs at alpha=0.5',
                            '3': '#MAs at alpha=0.6',
                            '4': '#MAs at alpha=0.7',
                            '5': '#MAs at alpha=0.8',
                            '6': '#MAs at alpha=0.9',
                            '7': '#MAs at alpha=1',
                            '8': '#Exudates 1',
                            '9': '#Exudates 2',
                            '10': '#Exudates 3',
                            '11': '#Exudates 4',
                            '12': '#Exudates 5',
                            '13': '#Exudates 6',
                            '14': '#Exudates 7',
                            '15': '#Exudates 8',
                            '16': 'euclidean distance of the center of the macula and the center of the optic disc',
                            '17': 'diameter of the optic disc',
                            '18': 'AM/FM Classification'}, inplace=True)
# diabetes_df

# Configure & store data
Y = diabetes_df['AM/FM Classification']
X = diabetes_df.drop('AM/FM Classification', axis=1).drop('Class', axis=1)
store_data(X, Y)

## 2. Default of credit card clients
# Import data
credit_df = pd.read_csv('datasets/defaultofcreditcardclients.csv', sep=',', dtype=int)
# credit_df

# Configure & store data
Y = credit_df['default payment next month']
X = credit_df.drop('default payment next month', axis=1).drop('ID', axis=1)
store_data(X, Y)

## 3. Breast Cancer Wisconsin
# .names has 11 cols
names = ["ID",
         "Clump Thickness",
         "Uniformity of Cell Size",
         "Uniformity of Cell Shape",
         "Marginal Adhesion",
         "Single Epithelial Cell Size",
         "Bare Nuclei",
         "Bland Chromatin",
         "Normal Nucleoli",
         "Mitoses",
         "Class"]
cancer_df = pd.read_csv("datasets/breast-cancer-wisconsin.data", names=names)
# cancer_df

# Configure & store data
Y = cancer_df['Class']
X = cancer_df.drop('Class', axis=1).drop('ID', axis=1)
X = X.replace('?', np.nan)
store_data(X, Y)

## 4. Statlog (German credit data)
# Import data. Assume last col is assessment.
statlog_df = pd.read_csv('datasets/german.data-numeric', delim_whitespace=True, header=None)
# statlog_df

# Configure & store data
Y = statlog_df[24]
X = statlog_df.drop(24, axis=1)
store_data(X, Y)

## 5. Adult
# Import data
adult_df = pd.read_csv('datasets/adult.data', header=None)
# Convert word data into cols of yes/no
adult_df = pd.get_dummies(adult_df)
# adult_df

# Configure & store data
Y = adult_df['14_ >50K']
X = adult_df.drop('14_ >50K', axis=1).drop('14_ <=50K', axis=1)
store_data(X, Y)

## 6. Yeast
# Import data
yeast_df = pd.read_csv('datasets/yeast.data', delim_whitespace=True, header=None)
# Ignore Sequence Name as it's unique
yeast_df = yeast_df.drop(0, axis=1)
# Convert word data into cols of yes/no
yeast_df = pd.get_dummies(yeast_df)
# yeast_df

# Will classify if Cystolic or Not (Including others falls under regression)
Y = yeast_df['9_CYT']
X = yeast_df.iloc[:, :8]
store_data(X, Y)

## 7. Thoracic Surgery Data
# Import data
raw_data = loadarff('datasets/ThoraricSurgery.arff')
thoracic_df = pd.DataFrame(raw_data[0])
# Don't want to run get_dummies on T/F since it would double it's weight
thoracic_df = thoracic_df.replace([b'F'], '0').replace([b'T'], '1')
# thoracic_df

# Configure & store data
Y = thoracic_df['Risk1Yr']
thoracic_df = thoracic_df.drop('Risk1Yr', axis=1)
X = pd.get_dummies(thoracic_df)
store_data(X, Y)

## 8. Seismic-Bumps
# Import data
raw_data = loadarff('datasets/seismic-bumps.arff')
seismic_df = pd.DataFrame(raw_data[0])
seismic_df = seismic_df.replace([b'0'], '0').replace([b'1'], '1')
seismic_df

# Configure & store data
Y = seismic_df['class']
seismic_df = seismic_df.drop('class', axis=1)
X = pd.get_dummies(seismic_df)
store_data(X, Y)


# Models used for testing. Re-initialized for each dataset
def generate_models():
    # Generate and store default models
    classification_models = []
    # Logistic Regression
    classification_models.append(LogisticRegression(fit_intercept=False, random_state=0))
    # SVM
    classification_models.append(LinearSVC(C=100, random_state=0))
    # Decision Tree
    classification_models.append(DecisionTreeClassifier(max_depth=100, random_state=0))
    # Random Forest
    classification_models.append(RandomForestClassifier(random_state=0))
    # K-nearest Neighbors
    classification_models.append(KNeighborsClassifier(n_neighbors=8))
    # AdaBoost
    classification_models.append(AdaBoostClassifier(random_state=0, n_estimators=100))
    # Gaussian Naive Bayes
    classification_models.append(GaussianNB())
    # Neural Network
    classification_models.append(MLPClassifier(random_state=0))

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
        results[index_src * 8 + index] = model.score(X_testing_sets[index_src], y_testing_sets[index_src])
print("Done")


######## DISPLAY RESULTS ########

results = results.reshape(8, 8)
print('\t', end='\t')
print(*MODEL_TYPES, sep='\t')
for index, row in enumerate(results):
    print(DATA_SOURCES[index], end='\t')
    print(*row, sep='\t')
