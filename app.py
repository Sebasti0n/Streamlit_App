import streamlit as st
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from itertools import combinations
from sklearn.metrics import mean_squared_error, r2_score
from pycaret.regression import *  #import from https://pycaret.gitbook.io/docs/

# Streamlit webpage title
st.title('Loss Function')

# Assume 'f' and 'dfdx' are defined somewhere in your script
# ...

# Streamlit sidebar widgets for user input
R = st.sidebar.slider('Rate of Maximum Population Growth (R)', 0.1, 2.0, 1.0)
X_BOUNDARY = st.sidebar.slider('Boundary Condition Coordinate (X_BOUNDARY)', -1.0, 1.0, 0.0)
F_BOUNDARY = st.sidebar.slider('Boundary Condition Value (F_BOUNDARY)', 0.0, 1.0, 0.5)

# Function to create the loss function
def make_loss_fn(f: Callable, dfdx: Callable) -> Callable:
    def loss_fn(params: torch.Tensor, x: torch.Tensor):
        # interior loss
        f_value = f(x, params)
        interior = dfdx(x, params) - R * f_value * (1 - f_value)

        # boundary loss
        x0 = X_BOUNDARY
        f0 = F_BOUNDARY
        x_boundary = torch.tensor([x0])
        f_boundary = torch.tensor([f0])
        boundary = f(x_boundary, params) - f_boundary

        loss = nn.MSELoss()
        loss_value = loss(interior, torch.zeros_like(interior)) + loss(
            boundary, torch.zeros_like(boundary)
        )

        return loss_value

    return loss_fn

# Assuming you have a mechanism to get 'params' and 'x'
# params = ...
# x = ...
# loss_fn = make_loss_fn(f, dfdx)
# loss_value = loss_fn(params, x)
# st.write('Loss Value:', loss_value)

