#RMSE (Root Mean Squared Error) is a metric that tells us how far apart our predicted values are from the actual values.
#It is calculated as the square root of the average of the squared differences between the predicted and actual values.
#The formula for RMSE is:  RMSE = sqrt(1/n * sum((predicted - actual)^2))
#Where n is the number of data points, predicted is the predicted value, and actual is the actual value.
#The lower the RMSE, the better the model.

import numpy as np

def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)
    n = len(pred)
    rmse = np.sqrt(np.sum((pred - tar) ** 2) / n)
    return rmse

# Example usage:
predictions = [3, -0.5, 2, 7]
targets = [2.5, 0.0, 2, 8]
print(f"The RMSE is: {rmse(predictions, targets)}")
# The RMSE is: 0.6123724356957945