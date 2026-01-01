# module1 task1

Penguin Dataset – Exploratory Data Analysis
Overview

This notebook explores the penguin classification dataset to better understand the structure of the data before building a machine learning model.

Dataset Loading

The dataset is loaded using pandas:

penguins = pd.read_csv("scikit-learn-mooc/datasets/penguins_classification.csv")

Data Inspection

We inspect the dataset to understand its structure:

penguins.head()
penguins.dtypes
penguins.columns

Target Distribution

We check how many samples belong to each penguin species:

penguins["Species"].value_counts()


This helps identify any class imbalance.

Feature Visualization

We visualize the distributions of numerical features:

penguins[["Culmen Length (mm)", "Culmen Depth (mm)"]].hist(figsize=(8, 4))

Pairwise Feature Relationships

We explore how features relate to each other using a pair plot:

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(
    penguins,
    vars=["Culmen Length (mm)", "Culmen Depth (mm)"],
    hue="Species"
)
plt.show()

Summary

The dataset contains multiple penguin species.

Numerical features show clear patterns between species.

Visualization helps assess separability before modeling.

# module1 task2

K-Nearest Neighbors – Adult Census Dataset
Overview

This notebook uses the Adult Census numeric dataset to train and evaluate a K-Nearest Neighbors (KNN) classifier.
The goal is to observe model performance on both training and test datasets.

Dataset Loading

The dataset is loaded from CSV files:

adult_census = pd.read_csv("scikit-learn-mooc/datasets/adult-census-numeric.csv")


The target variable is class, and all other columns are used as features.

data = adult_census.drop(columns="class")
target = adult_census["class"]

Dataset Path Check

To ensure the dataset exists and is correctly located:

from pathlib import Path
import os

print("CWD:", os.getcwd())
print("Exists datasets?", Path("datasets").exists())
print("Exists file?", Path("datasets/adult-census-numeric.csv").exists())
print("Found:", list(Path().glob("**/adult-census-numeric.csv"))[:5])

Model: K-Nearest Neighbors

We use a K-Nearest Neighbors classifier with k = 50:

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=50)
model.fit(data, target)

Training Accuracy

We evaluate the model on the training dataset:

train_accuracy = model.score(data, target)


This gives an idea of how well the model fits the training data.

Testing on New Data

We evaluate the model on a separate test dataset:

test_data = pd.read_csv("scikit-learn-mooc/datasets/adult-census-numeric-test.csv")

X_test = test_data.drop(columns="class")
y_test = test_data["class"]

test_accuracy = model.score(X_test, y_test)

Summary

A KNN classifier was trained using numeric features only.

Training accuracy shows how well the model fits the training data.

Test accuracy evaluates generalization on unseen data.

Using a large number of neighbors (k=50) helps reduce overfitting.

# module1 task3

Adult Census – Baseline Model (Dummy Classifier)
Overview

This notebook builds a simple baseline model using the Adult Census dataset.
The goal is to evaluate a very naive classifier before using more advanced models.

Dataset Loading

The dataset is loaded from a CSV file:

adult_census = pd.read_csv("scikit-learn-mooc/datasets/adult-census-numeric.csv")


The target variable is class, which represents income category.

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)

Feature Selection

Only numerical features are selected:

numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
data_numeric = data[numerical_columns]

Train / Test Split

The dataset is split into training and testing sets:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data,
    target,
    test_size=0.1,
    random_state=42
)

Baseline Model: DummyClassifier

A DummyClassifier is used to create a simple baseline model that always predicts the same class (>50K).

from sklearn.dummy import DummyClassifier

dummy50k = DummyClassifier(strategy="constant", constant=" >50K")
dummy50k.fit(X_train, y_train)

Model Evaluation

We evaluate the model using accuracy on the test set:

accuracy50k = dummy50k.score(X_test, y_test)


This accuracy serves as a baseline to compare against more advanced models.

Target Inspection

We check the unique values of the target column:

target.unique()


And clean potential whitespace:

target = target.str.strip()

Summary

A baseline classifier was built using DummyClassifier.

The model always predicts the same income class.

This provides a reference score to compare with more complex models.

Any useful model should outperform this baseline.

# module1 task4
Logistic Regression – Categorical Encoding & Cross-Validation
Overview

This notebook explores the use of Logistic Regression with different encoding strategies for categorical features.
We compare One-Hot Encoding and Ordinal Encoding, and evaluate model performance using cross-validation.

Dataset Preparation

We start by loading the dataset and separating features from the target.

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])

Selecting Categorical Features

We identify categorical columns using make_column_selector:

from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)

data_categorical = data[categorical_columns]

Models Definition
1. Logistic Regression with One-Hot Encoding
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    LogisticRegression(max_iter=200)
)


This approach creates binary features for each category and works well for linear models.

2. Logistic Regression with Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder

model1 = make_pipeline(
    OrdinalEncoder(),
    LogisticRegression()
)


This approach assigns integer values to categories, which may introduce artificial ordering.

Cross-Validation Example

To evaluate the model performance, we use cross-validation:

from sklearn.model_selection import cross_validate
from sklearn.datasets import make_classification

data, target = make_classification(
    n_samples=100,
    n_features=5,
    random_state=42
)

model = LogisticRegression()
scores = cross_validate(model, data, target)

print("The mean cross-validation accuracy is:", scores["test_score"].mean())

Summary

OneHotEncoder is usually preferred for categorical variables in linear models.

OrdinalEncoder may introduce unwanted ordinal relationships.

Cross-validation provides a reliable estimate of model performance.

Logistic Regression is a strong baseline model for classification tasks.



# # module1 task5 

Adult Census – Tree-Based Models & Feature Preprocessing
Overview

This notebook evaluates the impact of feature preprocessing when using a tree-based model (HistGradientBoostingClassifier) on the Adult Census dataset.

We compare:

Ordinal encoding only

Ordinal encoding + numerical scaling

Dataset Loading
adult_census = pd.read_csv("scikit-learn-mooc/datasets/adult-census.csv")


Target and features:

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])

Feature Selection

Categorical and numerical columns are identified automatically:

from sklearn.compose import make_column_selector as selector

numerical_columns = selector(dtype_exclude=object)(data)
categorical_columns = selector(dtype_include=object)(data)

Experiment 1 — Ordinal Encoding (No Scaling)

We encode categorical variables using OrdinalEncoder and leave numerical features unchanged.

from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_validate
import time

categorical_preprocessor = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)

preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    remainder="passthrough",
)

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

start = time.time()
cv_results = cross_validate(model, data, target)
elapsed_time = time.time() - start

scores = cv_results["test_score"]

print(
    f"The mean cross-validation accuracy is: "
    f"{scores.mean():.3f} ± {scores.std():.3f} "
    f"with a fitting time of {elapsed_time:.3f} seconds"
)

Experiment 2 — Ordinal Encoding + Scaling Numerical Features

Here we additionally apply StandardScaler to numerical features.

from sklearn.preprocessing import StandardScaler

categorical_preprocessor = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)

numerical_preprocessor = StandardScaler()

preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    (numerical_preprocessor, numerical_columns),
)

model2 = make_pipeline(
    preprocessor,
    HistGradientBoostingClassifier()
)

start = time.time()
cv_results = cross_validate(model2, data, target)
elapsed_time = time.time() - start

scores = cv_results["test_score"]

print(f"The mean cross-validation accuracy is: {scores.mean():.3f}")

Summary

Ordinal encoding alone already performs well with tree-based models.

Adding feature scaling does not significantly improve performance for HistGradientBoostingClassifier.

Tree-based models are generally insensitive to feature scaling.

Cross-validation is used to ensure stable performance estimates.

# module2 task1
Blood Transfusion Dataset – SVM Model Analysis
Overview

This notebook explores the use of a Support Vector Machine (SVM) classifier on the Blood Transfusion dataset.
We evaluate the model using cross-validation, study the effect of the gamma hyperparameter, and analyze model behavior using a learning curve.

Dataset Loading
import pandas as pd

blood_transfusion = pd.read_csv("scikit-learn-mooc/datasets/blood_transfusion.csv")
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]

Model Definition

We use a Support Vector Machine with:

StandardScaler for feature scaling

RBF kernel (default in SVC)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf")
)

Cross-Validation (Generalization Performance)

We evaluate the model using ShuffleSplit cross-validation.

from sklearn.model_selection import ShuffleSplit, cross_validate

cv = ShuffleSplit(random_state=0)

scores = cross_validate(
    model,
    data,
    target,
    cv=cv,
    scoring="accuracy"
)["test_score"]

print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")


This gives an estimate of how well the model generalizes to unseen data.

Validation Curve (Effect of Gamma)

We analyze how the gamma parameter affects model performance.

import numpy as np
from sklearn.model_selection import ValidationCurveDisplay

gamma_range = np.logspace(-3, 2, num=30)

ValidationCurveDisplay.from_estimator(
    model,
    data,
    target,
    param_name="svc__gamma",
    param_range=gamma_range,
    cv=cv,
)


This plot shows:

Training accuracy

Validation accuracy
for different values of gamma.

Learning Curve

We evaluate whether adding more training samples improves performance.

from sklearn.model_selection import LearningCurveDisplay

LearningCurveDisplay.from_estimator(
    model,
    data,
    target,
    cv=cv,
)


This helps determine:

Underfitting vs overfitting

Whether collecting more data would be beneficial

Summary

SVM with RBF kernel performs well on the dataset.

Cross-validation gives a reliable estimate of generalization performance.

Gamma strongly affects model flexibility.

Learning curves help assess whether more data would improve results.

# moudle3 task1
Adult Census – Hyperparameter Tuning with HistGradientBoosting
Overview

This experiment explores how different hyperparameters affect the performance of a HistGradientBoostingClassifier on the Adult Census dataset.
We perform a manual grid search using nested loops and evaluate each configuration using cross-validation.

Dataset Preparation

We load the dataset and separate features from the target variable.

adult_census = pd.read_csv("scikit-learn-mooc/datasets/adult-census.csv")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])

Train / Test Split

We split the dataset into training and testing subsets:

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, train_size=0.2, random_state=42
)

Preprocessing

Categorical features are encoded using OrdinalEncoder.

from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder

categorical_preprocessor = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)

preprocessor = make_column_transformer(
    (categorical_preprocessor, selector(dtype_include=object)),
    remainder="passthrough",
)

Model Definition

We use a HistGradientBoostingClassifier, which is efficient and works well with large datasets.

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(random_state=42)),
    ]
)

Hyperparameter Search (Manual Grid Search)

We test combinations of:

learning_rate: [0.01, 0.1, 1, 10]

max_leaf_nodes: [3, 10, 30]

Cross-validation is used to evaluate each configuration.

import numpy as np
from sklearn.model_selection import cross_val_score

learning_rates = [0.01, 0.1, 1, 10]
max_leaf_nodes_list = [3, 10, 30]

best_score = -np.inf
best_params = None

for lr in learning_rates:
    for mln in max_leaf_nodes_list:
        model.set_params(
            classifier__learning_rate=lr,
            classifier__max_leaf_nodes=mln
        )
        scores = cross_val_score(model, data_train, target_train, cv=5)
        mean_score = scores.mean()

        print(f"lr={lr}, max_leaf_nodes={mln} -> CV acc={mean_score:.3f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = {"learning_rate": lr, "max_leaf_nodes": mln}

Best Model
print("Best CV score:", round(best_score, 3))
print("Best params:", best_params)


This identifies the best hyperparameter combination based on cross-validation accuracy.

Summary

A manual grid search was performed using nested loops.

Cross-validation ensures robust performance estimation.

learning_rate and max_leaf_nodes significantly impact model performance.

The best-performing configuration can be used for final training and evaluation.
