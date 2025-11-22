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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def sanitize_filename(name):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name)


def create_output_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def get_faculty_meta():
    resp = requests.get(FACULTY_LIST_URL, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    faculty = []

    items = soup.find_all("div", class_="faculty-box-item")
    for item in items:
        name_tag = item.find("h5")
        link_tag = name_tag.find("a") if name_tag else None
        dept_tag = item.find("h6", class_="primary-color")

        if not link_tag:
            continue

        url = urljoin(BASE_URL, link_tag["href"])
        name = link_tag.get_text(strip=True)

        block_text = item.get_text(separator="\n", strip=True).split("\n")
        school = ""
        for line in block_text:
            if line.startswith("School of"):
                school = line.strip()
                break

        department = dept_tag.get_text(strip=True) if dept_tag else ""

        faculty.append(
            {"url": url, "name": name, "school": school, "department": department}
        )

    return faculty


def extract_faculty_text(url):
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    content = []

    accordions = soup.find_all("div", class_="accordion")
    for acc in accordions:
        for h2 in acc.find_all("h2"):
            title = h2.get_text(strip=True)
            cont = h2.find_next("div", class_="accordion-collapse")
            if cont:
                text = cont.get_text(separator="\n", strip=True)
                content.append(f"{title}:")
                content.append(text)
                content.append("")

    result = "\n".join(content)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def save_text(text, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    create_output_directory()
    faculty = get_faculty_meta()

    for i, entry in enumerate(faculty, 1):
        print(f"[{i}/{len(faculty)}] {entry['url']}")

        profile_text = extract_faculty_text(entry["url"])

        merged = "\n".join(
            [entry["name"], entry["school"], entry["department"], "", profile_text]
        )

        slug = entry["url"].rstrip("/").split("/")[-1]
        fname = sanitize_filename(slug) + ".txt"

        save_text(merged, fname)
        time.sleep(1)

    print("done.")


if __name__ == "__main__":
    main()
