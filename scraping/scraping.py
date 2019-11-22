import time

import pandas as pd

import requests
from bs4 import BeautifulSoup

with open("./scraping_result.txt", "w", encoding="UTF-8") as f:
    df = pd.read_csv("./20191101_samurai_article_categories.csv")
    url_list = df["URL"].tolist()

    for url in url_list:
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        div_toc_container = soup.find("div", id="toc_container")
        tokens = div_toc_container.get_text()
        f.write(url + " : " + tokens + "\n")
        time.sleep(1)
