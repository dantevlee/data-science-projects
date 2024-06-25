from bs4 import BeautifulSoup
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Function to scrape weather data
def scrape_weather_data():
    url = 'https://weather.com/weather/monthly/l/USNY0996:1:US'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Example: Extracting temperature data from table
    table = soup.find('table', {'class': 'twc-table'})
    rows = table.find_all('tr', {'data-head': 'Hi'})
    
    dates = []
    temperatures = []
    
    for row in rows:
        date = row.find('td', {'data-day': True}).text.strip()
        temperature = row.find('td', {'class': 'temp'}).text.strip()
        
        dates.append(date)
        temperatures.append(temperature)
    
    # Create a DataFrame
    df = pd.DataFrame({'Date': dates, 'Temperature': temperatures})
    return df

# Scrape data
weather_data = scrape_weather_data()
print(weather_data.head())


# Convert temperature to numerical
weather_data['Temperature'] = weather_data['Temperature'].str.replace('°', '').astype(int)

# Optional: Handle missing values if any

# Convert date to numerical for modeling (days since the start)
weather_data['Date'] = pd.to_datetime(weather_data['Date'])
weather_data['Days'] = (weather_data['Date'] - weather_data['Date'].min()).dt.days

# Prepare data for modeling
X = weather_data[['Days']].values
y = weather_data['Temperature'].values

# Create a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict temperatures for future dates
future_days = np.arange(weather_data['Days'].max() + 1, weather_data['Days'].max() + 31).reshape(-1, 1)
future_temperatures = model.predict(future_days)

# Visualize predictions


plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Temperatures')
plt.plot(future_days, future_temperatures, color='red', linestyle='--', label='Predicted Trend')
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.title('Temperature Trend Prediction')
plt.legend()
plt.grid(True)
plt.show()
