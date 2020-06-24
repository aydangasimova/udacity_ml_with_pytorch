# TODO: Add import statements
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv("../data/poly_reg.csv").sort_values(by="Var_X")
X = np.array(train_data[["Var_X"]])
y = train_data["Var_Y"].values

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the predictor feature
n = 4
poly_feat = PolynomialFeatures(n)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor features
poly_model = LinearRegression(fit_intercept = False).fit(X_poly, y)
poly_model.fit(X_poly,y)

plt.plot(X, poly_model.predict(X_poly))
plt.scatter(train_data['Var_X'], train_data['Var_Y'], zorder=3)
plt.show()
# Once you've completed all of the steps, select Test Run to see your model
# predictions against the data, or select Submit Answer to check if the degree
# of the polynomial features is the same as ours!