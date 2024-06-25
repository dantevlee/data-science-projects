import requests
from bs4 import BeautifulSoup
import pandas as pd
import re  # Import regex for pattern matching
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Function to scrape weather data from National Weather Service (NWS) website
def scrape_weather_data():
    url = 'https://forecast.weather.gov/MapClick.php?lat=37.7749&lon=-122.4194'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Initialize lists to store scraped data
    temperature_data = []
    humidity_data = []
    
    # Extract temperature data
    for temp in soup.find_all(class_='temp'):
        # Clean and extract numeric temperature values
        temp_value = re.findall(r'\d+', temp.text.strip())
        if temp_value:
            temperature_data.append(float(temp_value[0]))
        else:
            temperature_data.append(None)  # Handle missing or unexpected data
    
    # Extract humidity data
    for hum in soup.find_all(class_='humidity'):
        # Clean and extract numeric humidity values
        hum_value = re.findall(r'\d+', hum.text.strip())
        if hum_value:
            humidity_data.append(float(hum_value[0]))
        else:
            humidity_data.append(None)  # Handle missing or unexpected data
    
    # Ensure temperature and humidity data have the same length
    min_length = min(len(temperature_data), len(humidity_data))
    temperature_data = temperature_data[:min_length]
    humidity_data = humidity_data[:min_length]
    
    # Create a dataframe
    weather_df = pd.DataFrame({'Temperature (F)': temperature_data,
                               'Humidity (%)': humidity_data})
    
    # Drop rows with missing data
    weather_df.dropna(inplace=True)
    
    return weather_df

# Function to predict weather using RandomForestRegressor and plot results
def predict_weather(weather_df):
    X = weather_df[['Temperature (F)']]
    y = weather_df['Humidity (%)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    # Example prediction for a new temperature value
    new_temperature = [[70.0]]  # Example: Predict humidity for 70°F
    predicted_humidity = model.predict(new_temperature)
    print(f'Predicted Humidity: {predicted_humidity}')
    
    # Plotting predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual Humidity')
    plt.scatter(new_temperature, predicted_humidity, color='red', label='Predicted Humidity', marker='x', s=100)
    plt.plot(X_test, y_pred, color='green', linewidth=2, label='Regression Line')
    plt.xlabel('Temperature (F)')
    plt.ylabel('Humidity (%)')
    plt.title('Temperature vs. Humidity Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    weather_data = scrape_weather_data()
    print("Weather Data:")
    print(weather_data)
    
    predict_weather(weather_data)

if __name__ == '__main__':
    main()
