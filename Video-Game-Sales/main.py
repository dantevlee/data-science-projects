import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# Replace 'videogame_sales.csv' with your actual CSV file name and path
df = pd.read_csv('Video-Game-Sales\Vgsales.csv')

# Step 2: Inspect the first few rows of the dataframe
print(df.head())

# Step 3: Data Cleaning (if necessary)
# (Assume the data is already cleaned for this example)

# Step 4: Visualization
# Example visualization: Bar plot of top platforms by global sales
plt.figure(figsize=(10, 8))  # Increase the figure size (width, height)
top_platforms = df.groupby('Name')['Global_Sales'].first().nlargest(10).sort_values()
sns.barplot(x=top_platforms.values, y=top_platforms.index, palette='viridis', hue=top_platforms.index, dodge=False)
plt.xlabel('Global Sales (in millions)', fontsize=15)
plt.ylabel('Video Game', fontsize=15)
plt.title('Top 10 Video Games by Global Sales', fontsize=20)

# Adjust font size of y-axis labels
plt.yticks(fontsize=12)  # Change the font size as needed

plt.tight_layout()  # Ensures all elements fit within the figure area
plt.legend([], frameon=False)  # Hide the legend
plt.show()
