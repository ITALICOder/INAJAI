import random
import string
import os
import pandas as pd
import asyncio
import aiohttp
from bs4 import BeautifulSoup

async def generate_random_domain(min_length=2, max_length=10):
    """Generate a random domain name with a random length."""
    letters = string.ascii_lowercase
    length = random.randint(min_length, max_length)
    domain = ''.join(random.choice(letters) for _ in range(length)) + ".com"
    return domain

async def crawl_website(session, url):
    """Crawl a website and extract up to 5000 characters of HTML content."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Get text content from all paragraphs
                text = ' '.join(p.get_text() for p in soup.find_all('p'))
                return text[:5000]  # Return up to 5000 characters
    except Exception as e:
        return None  # Suppress error messages and return None

async def load_existing_data(file_path):
    """Load existing data from the CSV file if it exists."""
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        return existing_data, set(existing_data['website'].values)
    return pd.DataFrame(columns=['website', 'content']), set()

async def create_training_data(num_samples=100):
    """Create training data by generating random domains and crawling them."""
    csv_file_path = 'training_data/training_data.csv'
    
    # Load existing data and get already crawled domains
    existing_data, existing_domains = await load_existing_data(csv_file_path)
    
    # To store new data
    training_data = []

    async with aiohttp.ClientSession() as session:
        tasks = []

        for _ in range(num_samples):
            domain = await generate_random_domain()
            
            # Skip if domain already exists in the existing data
            if domain in existing_domains:
                continue

            tasks.append(crawl_website(session, f'http://{domain}'))

        # Gather results from all crawl tasks
        contents = await asyncio.gather(*tasks)

        for content in contents:
            if content:  # Only consider successful crawls
                domain_index = contents.index(content)  # Find domain index
                domain = await generate_random_domain()  # Get the corresponding domain
                print(f"Successfully crawled {domain}.")
                training_data.append({'website': domain, 'content': content})

    # Convert to DataFrame if there is any successful data
    if training_data:
        new_data_df = pd.DataFrame(training_data)

        if not os.path.exists('training_data'):
            os.makedirs('training_data')

        new_data_df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)
        print(f"Appended {len(training_data)} new entries to training_data/training_data.csv.")
    else:
        print("No successful crawls to save.")

async def main():
    counter = 0  # Initialize counter
    
    while True:
        counter += 1  # Increment counter
        await create_training_data(100)  # Generate training data for 100 random domains
        print(f"Round {counter}: Training data process completed.")

if __name__ == '__main__':
    asyncio.run(main())
