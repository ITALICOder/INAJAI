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
from tqdm import tqdm
from collections import Counter

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
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        async with session.get(url, headers=headers, timeout=25, allow_redirects=False) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                text = ' '.join(p.get_text() for p in soup.find_all('p'))
                return text[:5000], html
    except Exception: pass
    return None, None

async def load_existing_data(file_path: str) -> Tuple[pd.DataFrame, Set[str]]:
    """Load existing data from the CSV file if it exists."""
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        return existing_data, set(existing_data['website'].values)
    return pd.DataFrame(columns=['website', 'content']), set()

def generate_unique_domains(num_samples: int, existing_domains: Set[str]) -> List[str]:
    """Generate unique random domains based on national TLDs, avoiding existing domains."""
    target_count = len(NATIONAL_TLDS) * 10000  # Total domains to generate
    domains = set()
    
    while len(domains) < target_count:
        domain_length = random.randint(2, 33)
        domain_name = ''.join(random.choices(string.ascii_lowercase, k=domain_length))
        
        # Generate a domain for each TLD
        for tld in NATIONAL_TLDS:
            domain = f"{domain_name}{tld}"
            if domain not in existing_domains and domain not in domains:
                domains.add(domain)
                if len(domains) >= target_count:
                    break  # Stop if we've reached the target count
            else: break
    
    return list(domains)

def find_domains_in_content(html_content: str) -> Set[str]:
    """Extract new domains from HTML content using regex."""
    domain_pattern = r'https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    return set(re.findall(domain_pattern, html_content))

async def crawl_domains(session: aiohttp.ClientSession, domains: List[str], existing_domains: Set[str], concurrency_limit: int) -> Tuple[List[Dict[str, str]], Set[str]]:
    """Crawl domains with concurrency control."""
    successful_crawl_results = []
    new_domains = set()
    tld_counter = Counter()
    semaphore = asyncio.Semaphore(concurrency_limit)

    with tqdm(total=len(domains), desc="Crawling Progress", unit="request") as pbar:
        async def limited_crawl(domain):
            async with semaphore:
                if domain not in existing_domains:
                    content, html = await crawl_website(session, f'http://{domain}')
                    if content and html:
                        successful_crawl_results.append({'website': domain, 'content': content})
                        new_domains.update(find_domains_in_content(html))
                        tld = '.' + domain.split('.')[-1]
                        tld_counter[tld] += 1
                        existing_domains.add(domain)  # Add to existing set
                pbar.update(1)

        tasks = [limited_crawl(domain) for domain in domains]
        await asyncio.gather(*tasks)

    logger.info(f"Crawled {len(successful_crawl_results)} new domains successfully.")
    return successful_crawl_results, new_domains, tld_counter

async def create_training_data(session: aiohttp.ClientSession, num_samples: int = 10000):
    """Create training data by generating and crawling unique domains."""
    csv_file_path = 'training_data/training_data.csv'
    existing_data, existing_domains = await load_existing_data(csv_file_path)
    new_domains = generate_unique_domains(num_samples, existing_domains)
    logger.info(f"Generated {len(new_domains)} unique domains to crawl.")
    round_number=1
    while new_domains:      # Generate initial batch of unique domains
        logger.info(f"Starting a new round with {len(new_domains)} domains to crawl.")
        if round_number == 1: concurrency_limit = 150
        else:  concurrency_limit = 90
        training_data, found_domains, tld_counter = await crawl_domains(session, new_domains, existing_domains, concurrency_limit)

        if training_data:
            new_data_df = pd.DataFrame(training_data)
            existing_data = pd.concat([existing_data, new_data_df], ignore_index=True).drop_duplicates(subset='website', keep='last')
            existing_domains.update(new_data_df['website'])

            if not os.path.exists('training_data'):
                os.makedirs('training_data')

            existing_data.to_csv(csv_file_path, index=False, escapechar='\\')
            logger.info(f"Saved {len(training_data)} new entries. Total entries: {len(existing_data)}.")
        # Prepare for the next round with newly found domains not already in the dataset
        new_domains = [domain for domain in found_domains if domain not in existing_domains]
        logger.info(f"Round {round_number} Statistics:")
        if not new_domains:
            logger.info("No new domains found. Stopping the crawl.")
            break
        else:
            # Print statistics for this round
            logger.info(f" - Total successful crawls: {len(training_data)}")
            logger.info(f" - New domains found: {len(found_domains)}")
            logger.info(" - Top TLDs found:")
            for tld, count in tld_counter.most_common():
                logger.info(f"   {tld}: {count}")
        round_number += 1  # Increment round number
async def main():
    roundss=1
    while 1:
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            await create_training_data(session, num_samples=10000)
            logger.info(f"Completed all rounds(Nr. {roundss}) of domain crawling and training data update.")
            roundss += 1  # Increment round number

if __name__ == '__main__':
    asyncio.run(main())
