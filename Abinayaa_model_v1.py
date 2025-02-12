from ucimlrepo import fetch_ucirepo 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
  
# fetch dataset 
bcw_data = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = bcw_data.data.features 
y = bcw_data.data.targets 
  
# metadata 
print(bcw_data.metadata) 
  
# variable information 
print(bcw_data.variables) 

#To understand linear relationships between variables
correlation_matrix = X.corr().round(2)
plt.figure(figsize=(15,15))
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm',
center=0)
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.values.flatten()

# Initialize and train the classifier
clf = LogisticRegression(random_state=42, max_iter=10000)  #
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))