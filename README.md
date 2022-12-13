# Health-Care-Insurance-Cross-Selling-Prediction

This project has the purpose of determining which set of current customers are most likely to purchase vehicle insurance from our client (the health insurance company).
This is a final project for ADS-505: Business Analytics

Project Intro
Team
Abanather Negusu
Minsu Kim
Connie Chow
Methods Used
Data preprocessing
Exploratory Data Analysis
Data Visualization
Predictive Modeling
Performance Measures
Technologies & Platforms
Language: Python
Libraries: pandas, numpy, scikit learn, sklearn,
IDE: Jupyter Notebook
Project Description
The purpose of this project is to determine which set of customers will be most likely to purchase vehicle insurance through the current healthcare insurance provider.
An insurance policy is an arrangement by which a company undertakes to provide a guarantee for specified loss and/or damage. Healthcare insurance companies are always looking to expand their businesses and serve more customers.
By predicting if a customer is likely to churn, a business could prepare and implement strategies to reduce the churn process.
Additional outcomes of the project include determining what attributes of customers make them show a high likelihood of purchasing auto insurance when cross-sold to.
Project Requirements
data exploration/descriptive statistics
data processing/cleaning
statistical modeling
writeup/reporting
presentation
voice recordings
Required Python Packages
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from scipy import stats
from sklearn.cluster import KMeans
from pandas.api.types import CategoricalDtype
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import statsmodels.tools.tools as stattools
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import plot_tree
import random
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn import tree
Getting Started
Clone this repo using raw data.
add code and push new code into repo. To ensure cohesive code make sure to run all cells prior to upload.
Use ###### flags for output
Featured Notebooks/Analysis/Deliverables
[Presentation slides ](http link goes here)
Exploratory Data Analysis
Univariate Analysis

histograms showing distributions of values in each column
Multivariate Analysis

categorical columns plotted against target 'Response'
scatterplots
Data Pre-Processing
feature engineered by binning 'Age' and 'Vehicle_Age' - we wanted to see if there were higher response rates by the bin
dropped missing values
handled outliers - didn't drop them but kept and squished into the top 1 and bottom 3 percentile
Features Selection
Recursive Feature Elimination method
Random Forest Feature Importances
Pearson's high correlated features
Pearson's correlation feature to target
Modeling
Linear Models: Logistic Regression, Stochastic Gradient Descent
Ensemble Models: AdaBoost, RandomForest
Neural Networks: MLPClassifier (classifical neural network model)
General Models: Decision Tree
Evaluation - Performance Measures
All scores included, confusion matrices, reports, summaries
ROC Curve plotting performance of 5 models
Random Forest yielded best performance for prediction modeling
