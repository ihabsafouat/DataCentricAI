import os
import requests
from bs4 import BeautifulSoup

def scrape_pet_data(url: str, output_file: str) -> str:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')

    # Dummy logic: collect all paragraph text as data
    data = "\n".join(p.get_text() for p in soup.find_all('p'))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(data)

    return output_file
