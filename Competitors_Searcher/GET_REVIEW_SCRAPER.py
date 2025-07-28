import random
import time
import csv
import os
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium_stealth import stealth
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Set up Chrome options
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Uncomment for headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Initialize the webdriver
service = ChromeService(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Track visited URLs to prevent duplicates
visited_links = set()

# Check and load existing links from CSV to avoid re-saving them
output_filename = 'reviews.csv'
if os.path.exists(output_filename):
    existing_data = pd.read_csv(output_filename)
    existing_links = set(existing_data['company'].unique())
else:
    existing_links = set()

# Function to save data to CSV without duplicates
def save_to_csv(data, filename=output_filename):
    if not data:
        return
    
    new_data = pd.DataFrame(data)
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, new_data]).drop_duplicates(subset=['review', 'company'], keep='last')
    else:
        combined_df = new_data

    combined_df.to_csv(filename, index=False)

# Function to click an element safely
def safe_click(element):
    try:
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        time.sleep(1)
        element.click()
    except Exception as e:
        print(f"Error clicking element: {e}")

# Function to close popups if present
def close_popups():
    try:
        overlays = driver.find_elements(By.XPATH, "//div[contains(@class, 'popup') or contains(@class, 'overlay')]")
        for overlay in overlays:
            driver.execute_script("arguments[0].style.display = 'none';", overlay)
    except Exception as e:
        print(f"No overlays to close: {e}")
import time
import random
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Function to scrape reviews from a specific page
def scrape_reviews(link):
    if link in visited_links or link in existing_links:
        print(f"Skipping already visited link: {link}")
        return []

    visited_links.add(link)
    driver.get(link)
    time.sleep(random.uniform(2, 5))

    # Save the page source to an HTML file
    page_source_filename = f"page_source.html"
    with open(page_source_filename, 'w', encoding='utf-8') as file:
        file.write(driver.page_source)

    reviews = []
    try:
        close_popups()

        # Check for "No reviews" condition
        no_reviews_element = driver.find_elements(By.XPATH, "//span[@class='rating__reviews__count' and text()='0']")
        if no_reviews_element:
            reviews.append({
                'review': "No review found",
                'score': 0,
                'company': link.split("/")[-1]
            })
            return reviews

        # Parse the saved HTML file to extract reviews
        with open(page_source_filename, 'r', encoding='utf-8') as file:
            page_source = file.read()
            soup = BeautifulSoup(page_source, 'lxml')

            # Extract review elements based on the class structure
            review_elements = soup.find_all('div', class_='review__text')  # Change this based on the actual structure

            for element in review_elements:
                try:
                    # Extract the score from the 'data-rating' attribute
                    score = float(element['data-rating']) if 'data-rating' in element.attrs else 0

                    # Extract the review text within <p> tags
                    review_text = element.p.text.strip() if element.p else 'No review text'

                    reviews.append({
                        'review': review_text,
                        'score': score,
                        'company': link.split("/")[-1]
                    })
                except Exception as e:
                    print(f"Error extracting a review: {e}")

        # Try to find and click the "Next" button using JavaScript
        while True:
            try:
                next_button = driver.find_element(By.XPATH, "//a[contains(@class, 'pagination__button--enabled') and contains(text(), 'Next')]")
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(random.uniform(2, 5))
                with open(page_source_filename, 'w', encoding='utf-8') as file:
                    file.write(driver.page_source)

                # Repeat the review extraction process for the new page
                with open(page_source_filename, 'r', encoding='utf-8') as file:
                    page_source = file.read()
                    soup = BeautifulSoup(page_source, 'lxml')

                    # Extract review elements again
                    review_elements = soup.find_all('div', class_='review__text')  # Change this based on the actual structure

                    for element in review_elements:
                        try:
                            # Extract the score from the 'data-rating' attribute
                            score = float(element['data-rating']) if 'data-rating' in element.attrs else 0

                            # Extract the review text within <p> tags
                            review_text = element.p.text.strip() if element.p else 'No review text'

                            reviews.append({
                                'review': review_text,
                                'score': score,
                                'company': link.split("/")[-1]
                            })
                        except Exception as e: print(f"Error extracting a review: {e}")

            except Exception as e:
                print("No more pages or unable to click the next button: ", e)
                break  # Exit the while loop if there are no more pages

    except Exception as e: print(f"Error while scraping reviews: {e}")

    return reviews

# Function to extract review links from page source
def extract_review_links(page_source):
    pattern = r'href="/reviews/([a-zA-Z0-9.-]+)"'
    matches = re.findall(pattern, page_source)
    return [f"{base_url}/reviews/{match}" for match in matches]

# Main search and scraping function
def main_search_and_scrape(base_url):
    while True:
        random_length = random.randint(2, 7)
        random_query = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random_length))

        driver.get(f"{base_url}/search?q={random_query}")
        time.sleep(random.uniform(3, 6))

        page_source = driver.page_source
        review_links = extract_review_links(page_source)

        for href in review_links:
            reviews = scrape_reviews(href)
            save_to_csv(reviews)

        time.sleep(random.uniform(10, 30))

# Start the process
try:
    base_url = "https://www.sitejabber.com"
    main_search_and_scrape(base_url)
except KeyboardInterrupt: print("Scraping stopped by user.")
finally: driver.quit()
