# Housing Price Prediction with Decision Tree Regressor

This project demonstrates the implementation of a **Decision Tree Regressor** to predict housing prices based on features such as area, number of bedrooms, and number of bathrooms. The model is trained on a dataset, evaluated for accuracy, and provides predictions for both test data and user-defined inputs.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Code Explanation](#code-explanation)
- [Features](#features)
- [Evaluation Metrics](#evaluation-metrics)
- [How to Use](#how-to-use)
- [Output Example](#output-example)


## Prerequisites

- Python 3.7 or later
- Libraries:
  - `pandas`
  - `scikit-learn`

Install the required libraries using:

```bash
pip install pandas scikit-learn
```

## Setup and Installation

1. Clone the repository or download the script.
2. Ensure the dataset file `Housing.csv` is located in the `Data_Set` directory.
3. Run the script:

```bash
python DecisionTreeRegressor.py
```


## Code Explanation

### Data Loading

The script reads data from `Housing.csv` and provides a summary of its structure:

```python
data = pd.read_csv("Data_Set\\Housing.csv")
print(data.info())
```

### Data Preparation

- **Features (`x`)**: `area`, `bedrooms`, `bathrooms`
- **Target (`y`)**: `price`

Data is split into training (80%) and testing (20%) sets:

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### Model Training

A **Decision Tree Regressor** is created and trained:

```python
my_model = DecisionTreeRegressor()
my_model.fit(x_train, y_train)
```

### Prediction

- The model predicts prices for the test set:

```python
prediction = my_model.predict(x_test)
```

- It also predicts prices for user-defined input specifications:

```python
new_data = pd.DataFrame({"area": [1234], "bedrooms": [2], "bathrooms": [2]})
new_prediction = my_model.predict(new_data)
```

### Evaluation

The model's performance is evaluated using metrics like Mean Squared Error (MSE) and R² score:

```python
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)
```

## Features

- **Train-Test Split**: Splits the dataset for training and testing.
- **Custom Predictions**: Predicts prices for user-specified housing features.
- **Model Evaluation**: Measures performance using standard metrics.

## Evaluation Metrics

- **Mean Squared Error (MSE)**: Quantifies average error magnitude.
- **R² Score**: Proportion of variance explained by the model.

## How to Use

1. Run the script to view:
   - Dataset summary.
   - Predictions for the test set.
   - Predictions for user-defined housing specifications.
   - Model evaluation metrics.
2. Modify the `new_data` dictionary to test with different housing specifications.

## Output Example

### Test Data Predictions

```
Predicted prices for test data:
[450000. 300000. 600000.]
```

### Custom Input Prediction

```
User input/specification for the house:
   area  bedrooms  bathrooms
0  1234         2          2

Predicted price for the house: [275000.]
```

### Model Evaluation

```
Mean_Squared_Error:  1000000.0
R² score:  0.85
```
### Note:
```
Values may differ because, I considered random values here
```