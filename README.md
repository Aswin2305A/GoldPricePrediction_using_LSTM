# Gold Price Prediction using LSTM

A deep learning project that predicts gold prices using Long Short-Term Memory (LSTM) neural networks. The model analyzes historical gold price data from 2013-2023 to forecast future prices.

## Overview

This project implements a time series forecasting model using LSTM architecture to predict gold prices. The model uses a sliding window approach with 60 days of historical data to predict the next day's price.

## Dataset

- **Source**: Gold Price (2013-2023).csv
- **Time Period**: January 2013 - December 2022
- **Records**: 2,583 daily entries
- **Features**:
  - Date
  - Price (closing price)
  - Open
  - High
  - Low
  - Volume (dropped during preprocessing)
  - Change % (dropped during preprocessing)

## Model Architecture

The LSTM model consists of:

- **Input Layer**: Shape (60, 1) - 60 timesteps
- **LSTM Layer 1**: 64 units with return_sequences=True
- **Dropout Layer 1**: 0.2 dropout rate
- **LSTM Layer 2**: 64 units with return_sequences=True
- **Dropout Layer 2**: 0.2 dropout rate
- **LSTM Layer 3**: 64 units
- **Dropout Layer 3**: 0.2 dropout rate
- **Dense Layer**: 32 units with softmax activation
- **Output Layer**: 1 unit (predicted price)

**Total Parameters**: ~85,057

## Requirements

```python
numpy
pandas
matplotlib
plotly
scikit-learn
tensorflow
keras
```

## Installation

```bash
pip install numpy pandas matplotlib plotly scikit-learn tensorflow keras
```

## Usage

1. **Load the dataset**:
```python
df = pd.read_csv('Gold Price (2013-2023).csv')
```

2. **Run the notebook**:
   - Open `GoldPricePrediction_using_LSTM.ipynb` in Jupyter Notebook or Google Colab
   - Execute cells sequentially

## Data Preprocessing

1. **Date Conversion**: Convert date strings to datetime format
2. **Sorting**: Sort data chronologically
3. **Feature Selection**: Drop Volume and Change % columns
4. **Data Cleaning**: Remove commas from numeric values and convert to float
5. **Normalization**: Apply MinMaxScaler to scale prices between 0 and 1
6. **Train-Test Split**: 
   - Training data: 2013-2021
   - Test data: 2022 (260 days)
7. **Sequence Creation**: Create sliding windows of 60 days for input sequences

## Model Training

- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Epochs**: 30
- **Batch Size**: 32
- **Validation Split**: 10%

## Evaluation Metrics

The model is evaluated using:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R² Score** (Coefficient of Determination)

## Results

The model generates predictions for 2022 gold prices and visualizes:
- Training data (black line)
- Actual test data (blue line)
- Predicted test data (red line)

## Visualization

The project includes:
- Interactive plotly charts for exploratory data analysis
- Training vs test data split visualization
- Model performance comparison plot with actual vs predicted prices

## Project Structure

```
.
├── GoldPricePrediction_using_LSTM.ipynb    # Main notebook
├── Gold Price (2013-2023).csv              # Dataset
└── README.md                                # Project documentation
```

## Key Features

- Time series forecasting with LSTM
- Sliding window approach for sequence prediction
- Comprehensive data preprocessing pipeline
- Multiple evaluation metrics
- Visual comparison of predictions vs actual prices
- Dropout layers to prevent overfitting



## Acknowledgments

- Dataset source: Historical gold price data (2013-2023)
- Built with TensorFlow/Keras
