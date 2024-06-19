import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Function to fetch historical stock price data using Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    yf.pdr_override() # Activate Yahoo Finance API
    df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    return df

# Function to evaluate ARIMA model
def evaluate_arima_model(data, order):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    error = mean_squared_error(test, predictions)
    return error, predictions

# Main function to execute the stock price forecasting project
def main():
    # Fetch historical stock price data for Microsoft (symbol: MSFT) from Yahoo Finance
    symbol = 'AAPL'
    start_date = '2024-01-01'
    end_date = '2024-06-01'  # Adjust end date as needed
    df = fetch_stock_data(symbol, start_date, end_date)

    if df is not None:
        print(f"Successfully fetched data. Shape: {df.shape}")
        
        # Check for missing values
        print(df.isnull().sum())

        # Extract the closing price for analysis
        close_price = df['Close']

        # Plot the closing price over time
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, close_price, label='Closing Price', color='b')
        plt.title(f'{symbol} Closing Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Evaluate ARIMA models with different parameters
        order = (5, 1, 0)  # Example ARIMA parameters (p, d, q)
        mse, predictions = evaluate_arima_model(close_price.values, order)
        print(f'Mean Squared Error (ARIMA): {mse}')

        # Plot actual vs. predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[-len(predictions):], close_price.values[-len(predictions):], label='Actual', color='b')
        plt.plot(df.index[-len(predictions):], predictions, label='Predicted', color='r')
        plt.title(f'Actual vs. Predicted Closing Prices (ARIMA) for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print(f"Failed to fetch data.")

# Entry point of the script
if __name__ == "__main__":
    main()
