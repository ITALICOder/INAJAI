import os
import random
import re
import string
import pandas as pd
import asyncio
import aiohttp
import logging
from bs4 import BeautifulSoup
from typing import List, Tuple, Set, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
]
NATIONAL_TLDS = [
    '.com', '.net', '.org', '.info', '.biz',  # Generic TLDs
    '.uk',  # United Kingdom
    '.de',  # Germany
    '.fr',  # France
    '.jp',  # Japan
    '.au',  # Australia
    '.ca',  # Canada
    '.it',  # Italy
    '.es',  # Spain
    '.ru',  # Russia
    '.nl',  # Netherlands
    '.br',  # Brazil
    '.cn',  # China
    '.in',  # India
    '.mx',  # Mexico
    '.se',  # Sweden
    '.no',  # Norway
    '.fi',  # Finland
    '.dk',  # Denmark
    '.ch',  # Switzerland
    '.pl',  # Poland
    '.be',  # Belgium
    '.hk',  # Hong Kong
    '.sg',  # Singapore
    '.kr',  # South Korea
    '.za',  # South Africa
    '.at',  # Austria
    '.pt',  # Portugal
    '.ie',  # Ireland
    '.il',  # Israel
    '.nz',  # New Zealand
    '.tw',  # Taiwan
    '.th',  # Thailand
    '.ae',  # United Arab Emirates
    '.mys',  # Malaysia
    '.ph',  # Philippines
    '.vn',  # Vietnam
    '.ro',  # Romania
    '.by',  # Belarus
    '.hk',  # Hong Kong
    '.uy',  # Uruguay
    '.rs',  # Serbia
    '.sk',  # Slovakia
    '.bg',  # Bulgaria
    '.lt',  # Lithuania
    '.lv',  # Latvia
    '.ee',  # Estonia
    '.cy',  # Cyprus
    '.mt',  # Malta
    '.us',  # United States
    '.va',  # Vatican City
    '.ga',  # Gabon
    '.gl',  # Greenland
    '.np',  # Nepal
    '.am',  # Armenia
    '.kg',  # Kyrgyzstan
    '.kz',  # Kazakhstan
    '.uz',  # Uzbekistan
    '.tj',  # Tajikistan
    '.md',  # Moldova
    '.bn',  # Brunei
    '.cm',  # Cameroon
    '.ng',  # Nigeria
    '.ke',  # Kenya
    '.gh',  # Ghana
    '.tn',  # Tunisia
    '.dz',  # Algeria
    '.ma',  # Morocco
    '.jm',  # Jamaica
    '.tt',  # Trinidad and Tobago
    '.bb',  # Barbados
    '.bh',  # Bahrain
    '.qa',  # Qatar
    '.ao',  # Angola
    '.zw',  # Zimbabwe
    '.is',  # Iceland
    '.sm',  # San Marino
    '.li',  # Liechtenstein
    '.me',  # Montenegro
    '.mk',  # North Macedonia
    '.gs',  # South Georgia and the South Sandwich Islands
    '.um',  # United States Minor Outlying Islands
    '.pw',  # Palau
    '.as',  # American Samoa
    '.fm',  # Federated States of Micronesia
    '.cc',  # Cocos (Keeling) Islands
    '.tv',  # Tuvalu
    '.ws',  # Samoa
    '.bz',  # Belize
    '.bt',  # Bhutan
    '.cv',  # Cape Verde
    '.cf',  # Central African Republic
    '.km',  # Comoros
    '.dj',  # Djibouti
    '.gn',  # Guinea
    '.ml',  # Mali
    '.sn',  # Senegal
    '.sd',  # Sudan
    '.sl',  # Sierra Leone
    '.st',  # São Tomé and Príncipe
    '.tz',  # Tanzania
    '.ug',  # Uganda
    '.np',  # Nepal
    '.eu',
]

async def crawl_website(session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
    """Crawl a website and extract up to 5000 characters of HTML content."""
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
    }
    try:
        async with session.get(url, headers=headers, timeout=15, allow_redirects=False) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                text = ' '.join(p.get_text() for p in soup.find_all('p'))
                return text[:5000], html  # Return content and HTML for further processing
            #else:
                #logger.warning(f"Received response status {response.status} for URL: {url}")
    except Exception as e: pass
        #logger.error(f"Error fetching {url}: {e}")
    return None, None

async def load_existing_data(file_path: str) -> Tuple[pd.DataFrame, Set[str]]:
    """Load existing data from the CSV file if it exists."""
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        return existing_data, set(existing_data['website'].values)
    return pd.DataFrame(columns=['website', 'content']), set()

def generate_unique_domains(num_samples: int = 1000) -> List[str]:
    """Generate unique random domains based on national TLDs."""
    domains = set()
    while len(domains) < num_samples:
        domain_length = random.randint(2, 14)
        domain_name = ''.join(random.choices(string.ascii_lowercase, k=domain_length))
        for tld in NATIONAL_TLDS:
            domains.add(f"{domain_name}{tld}")
    return list(domains)

def find_domains_in_content(html_content: str) -> Set[str]:
    """Extract new domains from HTML content using regex."""
    domain_pattern = r'https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    return set(re.findall(domain_pattern, html_content))

async def crawl_domains(session: aiohttp.ClientSession, domains: List[str], concurrency_limit: int = 100) -> Tuple[List[Dict[str, str]], Set[str]]:
    """Crawl domains with concurrency control."""
    successful_crawl_results = []
    new_domains = set()
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def limited_crawl(domain):
        async with semaphore:
            content, html = await crawl_website(session, f'http://{domain}')
            if content and html:
                successful_crawl_results.append({'website': domain, 'content': content})
                # Find new domains in the content
                found_domains = find_domains_in_content(html)
                new_domains.update(found_domains)

    tasks = [limited_crawl(domain) for domain in domains]
    await asyncio.gather(*tasks)

    return successful_crawl_results, new_domains

async def create_training_data(session: aiohttp.ClientSession, num_samples: int = 1000):
    """Create training data by generating domains based on national TLDs and crawling them."""
    csv_file_path = 'training_data/training_data.csv'
    existing_data, existing_domains = await load_existing_data(csv_file_path)
    
    new_domains = [domain for domain in generate_unique_domains(num_samples) if domain not in existing_domains]

    # Crawl the generated domains directly
    training_data, found_domains = await crawl_domains(session, new_domains)

    if training_data:
        new_data_df = pd.DataFrame(training_data)
        existing_data = pd.concat([existing_data, new_data_df], ignore_index=True)

        # Ensure no duplicates in existing_data
        existing_data = existing_data.drop_duplicates(subset='website', keep='last')

        if not os.path.exists('training_data'):
            os.makedirs('training_data')

        existing_data.to_csv(csv_file_path, mode='w', header=True, index=False)
        logger.info(f"Saved {len(training_data)} new entries to {csv_file_path}.")

        # Recursively call to crawl new domains found
        if found_domains:
            logger.info(f"Found {len(found_domains)} new domains, continuing the crawl.")
            await create_training_data(session, num_samples=len(found_domains))

async def main():
    async with aiohttp.ClientSession() as session:
        counter = 0
        while True:
            counter += 1
            await create_training_data(session, num_samples=1000)
            logger.info(f"Round {counter}: Training data process completed.")

if __name__ == '__main__':
    asyncio.run(main())

