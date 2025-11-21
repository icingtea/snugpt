import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin

BASE_URL = "https://snu.edu.in"
FACULTY_LIST_URL = f"{BASE_URL}/faculty/"
OUTPUT_DIR = os.path.join("data", "extracted", "faculty")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_-]+', '_', name)

def create_output_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def get_faculty_links():
    resp = requests.get(FACULTY_LIST_URL, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, 'html.parser')

    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('/faculty/') and href != '/faculty/' and href.count('/') >= 3:
            full = urljoin(BASE_URL, href)
            if full not in links:
                links.append(full)
    return links

def extract_faculty_text(url):
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, 'html.parser')

    data = []

    h1 = soup.find('h1')
    if h1:
        data.append("=" * 80)
        data.append(h1.get_text(strip=True))
        data.append("=" * 80)
        data.append("")

    subtitle = soup.find('h2')
    if subtitle:
        data.append(subtitle.get_text(strip=True))
        data.append("")

    accordions = soup.find_all('div', class_='accordion')
    for acc in accordions:
        for h2 in acc.find_all('h2'):
            title = h2.get_text(strip=True)
            cont = h2.find_next('div', class_='accordion-collapse')
            if cont:
                text = cont.get_text(separator="\n", strip=True)
                data.append(title)
                data.append("-" * 80)
                data.append(text)
                data.append("")

    full_text = "\n".join(data)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    return full_text.strip()

def save_text(text, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

def main():
    create_output_directory()
    links = get_faculty_links()

    for i, url in enumerate(links, 1):
        print(f"[{i}/{len(links)}] {url}")

        text = extract_faculty_text(url)

        slug = url.rstrip('/').split('/')[-1]
        fname = sanitize_filename(slug) + ".txt"

        save_text(text, fname)
        time.sleep(1)

    print("done.")

if __name__ == "__main__":
    main()
