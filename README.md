# LSTM Model with Harris Hawks Optimization for Time Series Forecasting

This repository provides a framework for using an LSTM (Long Short-Term Memory) neural network, optimized with the Harris Hawks Optimization (HHO) algorithm, to forecast time series data. This setup is flexible and allows users to input their own CSV datasets for forecasting a target variable of their choice.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The project aims to forecast a specified target variable in time series data using an LSTM neural network, a powerful tool for capturing temporal dependencies. To enhance the model's performance, the Harris Hawks Optimization (HHO) algorithm is used to fine-tune hyperparameters such as the number of LSTM units and batch size.

## Dataset

Users can utilize their own dataset in CSV format. The dataset should include:

- A `datetime` column to be used as the index.
- One or more feature columns that provide data related to the target variable.
- A target variable column that you wish to predict. The name of this column can be customized in the code.

Ensure that your dataset is properly preprocessed, including handling any missing values and scaling numerical features if necessary.

## Installation

To set up the environment for this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
4. **install the required packages:**
   ```bash
   pip install -r requirements.txt

## Usage

To use this code with your own dataset, perform the following steps:

1. **Prepare your dataset**:
   - Ensure your CSV file has a `datetime` column and relevant feature and target variable columns.
   - Place your CSV file in the appropriate directory and note its path.

2. **Modify the `main.py` script**:
   - Update the path to your CSV file:
     ```python
     data = pd.read_csv('path/to/your/dataset.csv')
     ```
   - Set the `datetime` column as the index:
     ```python
     data['datetime'] = pd.to_datetime(data['datetime'])
     data.set_index('datetime', inplace=True)
     ```
   - Specify your feature columns and target variable:
     ```python
     feature_columns = ['YourFeature1', 'YourFeature2', ...]
     target_column = 'YourTargetVariable'
     ```
   - Ensure the target column is correctly specified in the `create_sequences` function:
     ```python
     y.append(data[target_column].iloc[i + n_steps])
     ```

3. **Run the script**:
   ```bash
   python main.py
4. **Check the output**:

   The script will output the best hyperparameters found by the HHO algorithm and display a plot comparing actual vs. predicted values.
