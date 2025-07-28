import os
import struct
import socket
import random
import time
import uuid
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Paths and settings
MODEL_PATH = 'duck_classifier.h5'
IMAGE_SAVE_DIR = 'data/duck_tiles'
PROXY_PORTS = list(range(9050, 9060))  # 9050–9059
SEARCH_URL = 'https://html.duckduckgo.com/html/?q=ip:{}'

# Ensure save directory
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Load the trained duck detector model
model = load_model(MODEL_PATH)

def is_duck_tile(img_path):
    """
    Load an image, preprocess, and predict with the duck model.
    Returns True if duck, False otherwise.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    pred = model.predict(np.expand_dims(img, 0))[0][0]
    return pred > 0.5


def setup_driver(socks_port):
    """
    Initialize a Chrome WebDriver instance using a given SOCKS5 proxy.
    """
    chrome_options = Options()
    chrome_options.add_argument(f'--proxy-server=socks5://127.0.0.1:{socks_port}')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    service = ChromeService()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def solve_captcha_and_collect(driver):
    """
    When DuckDuckGo presents the duck puzzle, capture each tile, save for training,
    classify with the model, and click the duck tiles automatically.
    """
    # Wait for puzzle to load
    time.sleep(2)

    # Locate all puzzle image tiles
    tiles = driver.find_elements(By.CSS_SELECTOR, 'div.anomaly-modal__puzzle img')
    tile_paths = []
    for idx, tile in enumerate(tiles):
        src = tile.get_attribute('src')
        # download or screenshot each tile
        tile_filename = os.path.join(IMAGE_SAVE_DIR, f'{uuid.uuid4().hex}.png')
        driver.get(src)
        driver.save_screenshot(tile_filename)
        tile_paths.append((tile, tile_filename))
        # navigate back to challenge page
        driver.back()

    # Classify and click
    for tile_el, img_path in tile_paths:
        if is_duck_tile(img_path): tile_el.click()

    # Submit the challenge
    submit = driver.find_element(By.CSS_SELECTOR, 'button.anomaly-modal__submit')
    submit.click()
    time.sleep(2)


def reverse_ip_selenium(ip):
    """
    Perform reverse IP lookup on DuckDuckGo via rotating proxies and Selenium.
    Returns a deduped list of domains.
    """
    domains = set()
    for port in PROXY_PORTS:
        driver = setup_driver(port)
        try:
            driver.get(SEARCH_URL.format(ip))
            time.sleep(2)

            # If captcha appears, solve it
            if driver.find_elements(By.CSS_SELECTOR, 'div.anomaly-modal__modal'): solve_captcha_and_collect(driver)

            # Parse results
            links = driver.find_elements(By.CSS_SELECTOR, 'a.result__a')
            for link in links:
                host = link.text.strip().split('/')[0]
                domains.add(host)

            # Break if we got results
            if domains:
                driver.quit()
                break
        except Exception as e: print(f"Error with proxy {port}: {e}")
        finally: driver.quit()
    return sorted(domains)


if __name__ == '__main__':
    while 1:
       TARGET_IP=socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
       found = reverse_ip_selenium(TARGET_IP)
       print(f"Domains for {TARGET_IP}: {found}")
       time.sleep(30)
