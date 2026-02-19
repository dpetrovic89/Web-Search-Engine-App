import requests
from bs4 import BeautifulSoup
import json
import os
import time
from urllib.parse import urljoin, urlparse
import hashlib

class Crawler:
    def __init__(self, seed_urls, max_pages=50, depth=2, output_dir="data"):
        self.seed_urls = seed_urls
        self.max_pages = max_pages
        self.depth = depth
        self.output_dir = output_dir
        self.visited = set()
        self.queue = []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def fetch(self, url):
        try:
            # More realistic user-agent
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                if 'text/html' in response.headers.get('Content-Type', ''):
                    return response.text
                else:
                    print(f"Skipping {url}: Not HTML ({response.headers.get('Content-Type')})")
            else:
                print(f"Error fetching {url}: Status {response.status_code}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return None

    def parse(self, html, url):
        soup = BeautifulSoup(html, 'html.parser')
        
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = soup.get_text(separator=' ', strip=True)
        title = soup.title.string if soup.title else url
        
        links = []
        domain = urlparse(url).netloc
        for a in soup.find_all('a', href=True):
            absolute_url = urljoin(url, a['href'])
            if urlparse(absolute_url).netloc == domain:
                links.append(absolute_url)
                
        return {
            "url": url,
            "title": title,
            "content": text,
            "links": list(set(links))
        }

    def save(self, data):
        url_hash = hashlib.md5(data['url'].encode()).hexdigest()
        filename = f"{url_hash}.json"
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved: {data['url']}")

    def crawl(self):
        for url in self.seed_urls:
            self.queue.append((url, 0))

        count = 0
        while self.queue and count < self.max_pages:
            url, current_depth = self.queue.pop(0)
            
            if url in self.visited or current_depth > self.depth:
                continue

            self.visited.add(url)
            print(f"[{count+1}/{self.max_pages}] Crawling: {url} (Depth: {current_depth})")
            
            html = self.fetch(url)
            if html:
                parsed_data = self.parse(html, url)
                if len(parsed_data['content']) > 100:
                    self.save(parsed_data)
                    count += 1
                    
                    for link in parsed_data['links']:
                        if link not in self.visited:
                            self.queue.append((link, current_depth + 1))
                else:
                    print(f"Skipping {url}: Low content")
            
            time.sleep(1)

if __name__ == "__main__":
    seeds = ["https://www.python.org/"]
    crawler = Crawler(seeds, max_pages=15)
    crawler.crawl()
