import csv
import re
import requests
from bs4 import BeautifulSoup
import time

def clean_price(price_text: str) -> int or None:
    """
    Cleans a price string (e.g., 'Tk 17,500') by removing currency and commas, 
    and converts it to an integer (17500).
    """
    if not price_text:
        return None
    # Remove 'Tk', commas, and any non-digit characters except for a period (though prices here seem integer)
    cleaned = price_text.replace('Tk', '').replace(',', '').strip()
    try:
        # Convert to integer
        return int(cleaned)
    except ValueError:
        return None

def scrape_bikroy_mobiles():
    """
    Scrapes mobile names and prices from Bikroy.com across pages 1 to 10.
    """
    # Base URL with a placeholder for the page number
    BASE_URL = "https://bikroy.com/en/ads/bangladesh/mobile-phones?sort=date&order=desc&buy_now=0&urgent=0&enum.condition=used&page="
    OUTPUT_FILE = 'bikroy_mobile_prices2.csv'
    all_results = []
    
    PAGE_START = 1
    PAGE_END = 20 # Scraping pages 1 through 10
    
    # Use a common browser User-Agent to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for page_num in range(PAGE_START, PAGE_END + 1):
        url = f"{BASE_URL}{page_num}"
        print(f"-> Scraping page {page_num}...")
        
        try:
            # 1. Fetch the page content
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status() # Check for HTTP errors
            
            # 2. Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 3. Find all ad content blocks using the specific class provided in the structure
            ad_blocks = soup.find_all('div', class_='content--3JNQz')

            if not ad_blocks:
                print(f"   Warning: No ads found on page {page_num}. Ending scraping process.")
                break
            
            page_data_count = 0
            for ad in ad_blocks:
                # 4. Extract Mobile Name from h2 with class 'title--3yncE'
                name_tag = ad.find('h2', class_='title--3yncE')
                mobile_name = name_tag.get_text(strip=True) if name_tag else "N/A"
                
                # 5. Extract Price from the span inside the div with class 'price--3SnqI'
                price_div = ad.find('div', class_='price--3SnqI')
                price_span = price_div.find('span') if price_div else None
                price_text = price_span.get_text(strip=True) if price_span else "N/A"
                
                # 6. Clean and convert the price to a number
                numerical_price = clean_price(price_text)
                
                all_results.append({
                    'Mobile Name': mobile_name,
                    'Price (Tk)': numerical_price,
                    'Original Price':''
                })
                page_data_count += 1
            
            print(f"   Successfully extracted {page_data_count} ads.")
            
        except requests.exceptions.RequestException as e:
            print(f"   Error fetching page {page_num}: {e}")
            
        # 7. Be polite: wait for 3 seconds before requesting the next page to avoid overloading the server.
        time.sleep(3) 

    # 8. Write all results to the CSV file
    if all_results:
        try:
            with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as outfile:
                fieldnames = ['Mobile Name', 'Price (Tk)', 'Original Price']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerows(all_results)
                
            print(f"\n✅ Scraping complete. Saved {len(all_results)} records to '{OUTPUT_FILE}'.")
        except IOError as e:
            print(f"\nError writing to output CSV file: {e}")
    else:
        print("\nNo data was successfully scraped.")

if __name__ == "__main__":
    scrape_bikroy_mobiles()