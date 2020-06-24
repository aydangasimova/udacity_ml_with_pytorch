# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv("../data/regularization_data.csv")
X = None
y = None

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = None

# TODO: Fit the model.


# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = None
print(reg_coef)