#import dependencies 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import streamlit as st
from itertools import combinations
from sklearn.metrics import mean_squared_error, r2_score
from pycaret.regression import *  #import from https://pycaret.gitbook.io/docs/

# Load data from the provided files 
file_paths = ['19A.npy', 'BT2.npy', 'F1B.npy', 'SR.npy']

data = {file_path.split('/')[-1].split('.')[0]: np.load(file_path) for file_path in file_paths}

data_prepared = {key: pd.DataFrame(data=value, columns=[f'feature_{i}' for i in range(value.shape[1] - 1)] + ['Vp']) for key, value in data.items()}

# Generating all possible combinations of three wells for training and one well for testing
combinations_list = list(combinations(data_prepared.keys(), 3))
#combinations_list = list(combinations(data_prepared.keys(), 3))


def plot_depth_predictions(depth, y_actual, y_predicted, title):
    ztop = np.min(depth) - 2.0
    zbot = np.max(depth)
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_actual, depth, '-', label='Measured', color='black')
    plt.plot(y_predicted, depth, '-', label='Predicted', color='r')
    plt.legend()
    plt.ylim(ztop, zbot)
    plt.gca().invert_yaxis()
    plt.xlabel("$V_p$ (Km/s)")
    plt.ylabel("Depth (m)")
    plt.title(title)
    plt.locator_params(axis='x', nbins=5)
    plt.xlim(1.5, 6)
    plt.grid(True)
    plt.show()
    
for combination in combinations_list:
    train_data = pd.concat([data_prepared[key].reset_index(drop=True) for key in combination], axis=0, ignore_index=True)
    test_data_key = list(set(data_prepared.keys()) - set(combination))[0]
    test_data = data_prepared[test_data_key]

    # Initialize PyCaret
    exp = setup(data=train_data, target='Vp', session_id=123)

    # Compare and create a model
    best_model = compare_models()

    # Train the best model
    trained_model = finalize_model(best_model)

    # Make predictions on the test data
    predictions = predict_model(trained_model, data=test_data)

    # Print the columns of the predictions DataFrame
    st.write("Predictions Columns:", predictions.columns)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(predictions['Vp'], test_data['Vp']))
    r2 = r2_score(predictions['Vp'], test_data['Vp'])
    
        # Print or log the results
    st.write(f"Combination: {combination}, Test Well: {test_data_key}")
    st.write(f"RMSE: {rmse}")
    st.write(f"R-squared: {r2}")

    # Plot predictions
    plot_depth_predictions(test_data.index, test_data['Vp'], predictions['Vp'], f"Prediction - {test_data_key}")