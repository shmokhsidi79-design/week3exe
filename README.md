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

