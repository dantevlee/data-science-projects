import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to scrape product data from Target
def scrape_target_products(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        products = soup.find_all('div', class_='ProductCard')

        product_data = []
        for product in products:
            try:
                name = product.find('a', class_='Link__StyledLink-kq1lha-1').text.strip()
                price = product.find('span', class_='styles__PriceFontSize-sc-1f5rw6x-8').text.strip()
                rating = product.find('div', class_='styles__RatingContainer-sc-6hlnhi-4').text.strip()
                
                product_data.append({'Name': name, 'Price': price, 'Rating': rating})
            except AttributeError:
                continue
        
        return product_data
    else:
        print(f'Failed to retrieve data from Target. Status code: {response.status_code}')
        return None

# Main function to scrape Target product data
def main():
    url = 'https://www.target.com/s?searchTerm=laptop'  # Example: Target URL for laptops
    product_data = scrape_target_products(url)
    
    if product_data:
        df = pd.DataFrame(product_data)
        print('Successfully scraped Target product data:')
        print(df.head())
        df.to_csv('target_products.csv', index=False)
        print('Saved data to target_products.csv')
    else:
        print('No product data found. Please check the website or try again later.')

if __name__ == '__main__':
    main()
