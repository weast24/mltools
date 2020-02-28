import time

import pandas as pd

import requests
from bs4 import BeautifulSoup


def scraping(input_filepath: str, output_filepath: str, selector: str) -> None:
    with open(output_filepath, "w", encoding="UTF-8") as f:
        df = pd.read_csv(input_filepath)
        url_list = df["URL"].tolist()

        for url in url_list:
            time.sleep(1)
            r = requests.get(url)
            soup = BeautifulSoup(r.content, "html.parser")
            els = soup.select(selector)
            for el in els:
                tokens = els.get_text()
                f.write(url + " : " + tokens + "\n")
