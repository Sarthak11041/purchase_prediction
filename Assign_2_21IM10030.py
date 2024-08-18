
#Name-Sarthak Gupta
#Roll No- 21IM10030

import numpy as np
import pandas as pd

df=pd.read_csv('dataset.csv')
df

df.describe(include='all').T

# Calculate the percentage of missing values in each column
missing_percentages = (df.isnull().sum() / len(df)) * 100

# Identify columns with missing values greater than 50%
columns_to_drop = missing_percentages[missing_percentages > 50].index

# Drop the identified columns from the DataFrame
df_cleaned = df.drop(columns=columns_to_drop)
df=df_cleaned
print(df_cleaned.shape)

columns = ['Product_Category_2']  # Replace with the actual column names
data_selected = df[columns]

# Calculate the mode of each selected column
column_modes = data_selected.mode().iloc[0]

# Replace NaN values in the selected columns with the column modes
data_selected_filled = data_selected.fillna(column_modes)
df.drop(['User_ID','Product_ID'], axis = 1, inplace = True)
# Replace the selected columns in the original DataFrame with the filled data
df[columns] = data_selected_filled
df.info()
df.describe(include='all').T

"""# **EDA**"""

# Commented out IPython magic to ensure Python compatibility.
import plotly
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

fig=px.histogram(df,x='Purchase',marginal='box',color='Product_Category_1',title='Purchase')
fig.update_layout(bargap=0.2)
fig.show()

fig=px.histogram(df,x='Purchase',marginal='box',color='Gender',title='Purchase')
fig.update_layout(bargap=0.05)
fig.show()

fig=px.histogram(df,x='Purchase',marginal='box',color='Age',title='Purchase')
fig.update_layout(bargap=0.2)
fig.show()

fig=px.histogram(df,x='Purchase',marginal='box',color='Stay_In_Current_City_Years',title='Purchase')
fig.update_layout(bargap=0.2)
fig.show()

fig=px.histogram(df,x='Purchase',marginal='box',color='Marital_Status',title='Purchase')
fig.update_layout(bargap=0.2)
fig.show()

df.nunique()

gender_codes = {'F':0, 'M':1}
df['Gender'] = df['Gender'].map(gender_codes)
df['Gender'].unique()

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(df[['City_Category']])
region_one_hot = enc.transform(df[['City_Category']])
region_one_hot
region_one_hot.toarray()
df[['A','B','C']] = region_one_hot.toarray()
df.drop(columns=['City_Category'], inplace=True)
df.info()

df = df.drop('C',axis = 1)

df_encoded = pd.get_dummies(df, columns=['Age', 'Occupation', 'Product_Category_1', 'Product_Category_2'])

df_encoded.info()

from sklearn.preprocessing import LabelEncoder
# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the column
df_encoded['Stay_In_Current_City_Years_encoded'] = label_encoder.fit_transform(df_encoded['Stay_In_Current_City_Years'])

# Drop the original column and rename the new one
df_encoded.drop(columns=['Stay_In_Current_City_Years','Age_55+','Occupation_20','Product_Category_1_20','Product_Category_2_18.0'], inplace=True)

df=df_encoded

df.info()

df.corr()

correlation_matrix = df.corr()

# Set the size of the heatmap
plt.figure(figsize=(70, 30))  # Adjust the width and height as needed

# Create the heatmap using Seaborn
sns.heatmap(correlation_matrix, cmap='Blues',annot=True,fmt=".3f")

# Show the plot
plt.show()

#Experiment 2



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Separate features and target variable
X = df.drop(columns=['Purchase'])  # Features
y = df['Purchase']  # Labels

# Apply feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training, validation, and testing sets (60:20:20)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Linear Regression using closed-form solution
class LinearRegressionClosedForm:
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Add a column of ones for the bias term
        self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.weights)

# Initialize and train the Linear Regression model using closed-form solution
model = LinearRegressionClosedForm()
model.fit(X_train, y_train)

# Make predictions on validation set
y_pred_validation = model.predict(X_validation)

# Make predictions on test set
y_pred_test = model.predict(X_test)

# Calculate Mean Squared Error for test set
mse_test = mean_squared_error(y_test, y_pred_test)
print("Test Mean Squared Error:", mse_test)

# Calculate Mean Absolute Percentage Error for test set
mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
print("Mean Absolute Percentage Error:", mape, "%")

X = df.drop(columns=['Purchase']) #features
y = df['Purchase'] #labels

# Separate features and target variable
X = df[['Stay_In_Current_City_Years_encoded']]
y = df['Purchase']

# Add a column of ones for the bias term
X_with_bias = np.c_[np.ones(X.shape[0]), X]

# Split the data into train, validation, and test sets in 60:20:20 ratio
X_train, X_temp, y_train, y_temp = train_test_split(X_with_bias, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Closed-form solution for Linear Regression
XTX = np.dot(X_train.T, X_train)
XTy = np.dot(X_train.T, y_train)
theta = np.dot(np.linalg.inv(XTX), XTy)

# Predictions on the validation set
y_val_pred = np.dot(X_val, theta)

# Predictions on the test set
y_test_pred = np.dot(X_test, theta)

# Calculate Mean Squared Error on test set
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"Mean Squared Error on Test Set: {mse_test:.5f}")

#Experiment 3



import torch
import torch.nn as nn
import torch.optim as optim

# Separate features and target variable
X = df.drop(columns=['Purchase']) #features
y = df['Purchase'] #labels

# Convert to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Split the data into train, validation, and test sets in 60:20:20 ratio
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Convert scaled features to PyTorch tensors
X_train_scaled_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_scaled_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

# Linear Regression model using PyTorch
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Learning rate values to test
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1]

# Lists to store learning rates and MSE values for plotting
lr_values = []
mse_values = []

# Training loop for different learning rates
num_epochs = 50
batch_size = 256

for lr in learning_rates:
    print(f"Learning Rate: {lr:.5f}")

    # Create the model instance
    input_size = X_train_scaled_tensor.shape[1]
    model = LinearRegressionModel(input_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        for i in range(0, len(X_train_scaled_tensor), batch_size):
            # Mini-batch data
            X_batch = X_train_scaled_tensor[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validate the model
    with torch.no_grad():
        y_val_pred = model(X_val_scaled_tensor)
        mse_val = mean_squared_error(y_val, y_val_pred)
        lr_values.append(lr)
        mse_values.append(mse_val)
        print(f"Mean Squared Error on Validation Set: {mse_val:.5f}")

# Plotting MSE vs Learning Rate
plt.plot(lr_values, mse_values, marker='o')
plt.xscale('log')  # Using a logarithmic scale for better visualization
plt.xlabel('Learning Rate (Log Scale)')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Learning Rate')
plt.grid(True)
plt.show()

#Experiment 4


import torch
import torch.nn as nn
import torch.optim as optim


X = df.drop(columns=['Purchase'])
y = df['Purchase']

# Convert to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Split the data into train, validation, and test sets in 60:20:20 ratio
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Convert scaled features to PyTorch tensors
X_train_scaled_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_scaled_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

# Ridge Regression model using PyTorch
class RidgeRegressionModel(nn.Module):
    def __init__(self, input_size, alpha):
        super(RidgeRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.alpha = alpha

    def forward(self, x):
        return self.linear(x)

# Hyperparameter values to test
alphas = np.arange(0.0, 1.1, 0.1)

# Lists to store alphas and MSE values for plotting
alpha_values = []
mse_values = []

# Training loop for different alphas
num_epochs = 50
batch_size = 256

for alpha in alphas:
    print(f"Alpha: {alpha:.2f}")

    # Create the model instance
    input_size = X_train_scaled_tensor.shape[1]
    model = RidgeRegressionModel(input_size, alpha)

    # Loss and optimizer with L2 regularization (weight decay)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=alpha)

    # Training loop
    for epoch in range(num_epochs):
        for i in range(0, len(X_train_scaled_tensor), batch_size):
            # Mini-batch data
            X_batch = X_train_scaled_tensor[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validate the model
    with torch.no_grad():
        y_val_pred = model(X_val_scaled_tensor)
        mse_val = mean_squared_error(y_val, y_val_pred)
        alpha_values.append(alpha)
        mse_values.append(mse_val)
        print(f"Mean Squared Error on Validation Set: {mse_val:.5f}")

# Plotting MSE vs Alpha
plt.plot(alpha_values, mse_values, marker='o')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Alpha (Regularization Strength)')
plt.grid(True)
plt.show()

#Experiment 5



# Separate features and target variable
X = df.drop(columns=['Purchase']) #features
y = df['Purchase'] #labels

# Convert to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Split the data into train and test sets in 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Scale the test set as well

# Convert scaled features to PyTorch tensors
X_train_scaled_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_scaled_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)  # Convert the test set to tensor

# Linear Regression model using PyTorch
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Learning rate to use
lr = 0.01

# Create the model instance
input_size = X_train_scaled_tensor.shape[1]
model = LinearRegressionModel(input_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Training loop
num_epochs = 50
batch_size = 256

for epoch in range(num_epochs):
    for i in range(0, len(X_train_scaled_tensor), batch_size):
        # Mini-batch data
        X_batch = X_train_scaled_tensor[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluate the model on test set
with torch.no_grad():
    y_test_pred = model(X_test_scaled_tensor)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"Mean Squared Error on Test Set: {mse_test:.5f}")

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np


X = df.drop(columns=['Purchase'])
y = df['Purchase']

# Convert to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Split the data into train, validation, and test sets in 60:20:20 ratio
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Apply feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Convert scaled features to PyTorch tensors
X_train_scaled_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_scaled_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

# Ridge Regression model using PyTorch
class RidgeRegressionModel(nn.Module):
    def __init__(self, input_size, alpha):
        super(RidgeRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.alpha = alpha

    def forward(self, x):
        return self.linear(x)

# Hyperparameter values
alpha = 0.0
learning_rate = 0.01
num_epochs = 50
batch_size = 256

# Create the model instance
input_size = X_train_scaled_tensor.shape[1]
model = RidgeRegressionModel(input_size, alpha)

# Loss and optimizer with L2 regularization (weight decay)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha)

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(X_train_scaled_tensor), batch_size):
        # Mini-batch data
        X_batch = X_train_scaled_tensor[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
with torch.no_grad():
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_pred = model(X_test_scaled_tensor)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"Mean Squared Error on Test Set: {mse_test:.5f}")
