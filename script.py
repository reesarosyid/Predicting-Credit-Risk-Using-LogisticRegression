# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle
warnings.filterwarnings('ignore')

# Gathering Data
df = pd.read_csv("modeldata.csv")

# Encoding
cat_cols = df.select_dtypes(include=["object","category"]).columns
encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# Seperate X and y label
X = df.drop('status', axis=1)
y = df['status']

# Split and scalling data
X_train, X_test, y_train, y_test = train_test_split(X , y, shuffle = True, test_size = 0.2, random_state = 42)

# Scaling data
scaler = MinMaxScaler()
scaler.fit(X_train)

# The transofrmation of X
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
lrmodel = LogisticRegression()
lrmodel.fit(X_train_scaled, y_train)
y_predLr = lrmodel.predict(X_test_scaled)

# Confussion Matrix
print(classification_report(y_test, y_predLr))

# Save the Model
model_filename = "LRmodel.pkl"

with open(model_filename, 'wb') as model_file:
    pickle.dump(lrmodel, model_file)
