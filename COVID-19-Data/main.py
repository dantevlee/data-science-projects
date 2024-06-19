import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter, MonthLocator

# Step 1: Download the dataset
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
response = requests.get(url)
data = StringIO(response.text)

# Step 2: Load the dataset into a DataFrame
df = pd.read_csv(data)

# Step 3: Inspect the first few rows of the dataframe
print(df.head())

# Step 4: Clean the data
# Drop unnecessary columns
df = df.drop(columns=['Province/State', 'Lat', 'Long'])

# Step 5: Reshape the DataFrame
# Convert the wide format into long format
df_long = pd.melt(df, id_vars=['Country/Region'], var_name='Date', value_name='Confirmed')

# Explicitly specify the date format
df_long['Date'] = pd.to_datetime(df_long['Date'], format='%m/%d/%y')

# Step 6: Filter for the year 2022 only
df_2022 = df_long[df_long['Date'].dt.year == 2022]

# Group by country and date to get the total confirmed cases per country
df_grouped = df_2022.groupby(['Country/Region', 'Date']).sum().reset_index()

# Step 7: Visualize the data
# Plot the top 5 countries with the highest number of confirmed cases
latest_date = df_grouped['Date'].max()
top_countries = df_grouped[df_grouped['Date'] == latest_date].nlargest(5, 'Confirmed')['Country/Region']
df_top_countries = df_grouped[df_grouped['Country/Region'].isin(top_countries)]

plt.figure(figsize=(14, 8))
sns.lineplot(data=df_top_countries, x='Date', y='Confirmed', hue='Country/Region')
plt.title('COVID-19 Confirmed Cases Over Time (2022)')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')

# Format the y-axis to display numbers in millions
formatter = FuncFormatter(lambda x, pos: f'{int(x / 1e6)}M')
plt.gca().yaxis.set_major_formatter(formatter)

# Format the x-axis to display dates in MM-YYYY format and show all months of 2022
date_formatter = DateFormatter("%m-%Y")
plt.gca().xaxis.set_major_formatter(date_formatter)
plt.gca().xaxis.set_major_locator(MonthLocator())

plt.legend(title='Country')
plt.xticks(rotation=45)
plt.show()
