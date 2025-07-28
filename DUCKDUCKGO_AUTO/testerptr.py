import requests
from bs4 import BeautifulSoup
import time

def reverse_ip_duckduckgo(ip, pause=1.0):
    """
    Scrape DuckDuckGo's HTML interface for `ip:<IP>` results.
    Returns a deduped list of domains.
    """
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0" 
    }
    params = {"q": f"ip:{ip}"}
    domains = set()

    # DuckDuckGo paginates; usually up to ~30 results first page
    resp = requests.post(url, data=params, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    print(resp)
    print(soup)
    for a in soup.select("a.result__a"):
        host = a.get_text(strip=True)
        # sometimes the link is a full URL; strip scheme/path
        host = host.split("/")[0]
        domains.add(host)

    # (Optional) follow “More Results” link
    more = soup.select_one("a.result--more__btn")
    if more:
        time.sleep(pause)
        href = more["href"]
        resp2 = requests.get("https://html.duckduckgo.com" + href,
                             headers=headers, timeout=10)
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        for a in soup2.select("a.result__a"):
            domains.add(a.get_text(strip=True).split("/")[0])

    return sorted(domains)

# Example
if __name__ == "__main__":
    for d in reverse_ip_duckduckgo("93.184.216.34"):
        print(d)
