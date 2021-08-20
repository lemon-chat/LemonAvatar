
import os
import shutil
import json
import time
import random
import datetime
from typing import Any, NoReturn, Optional, Union
from requests.sessions import session
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from lxml import etree
import urllib.parse
import urllib3
import time
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests
from selenium.webdriver.chrome.options import Options


class PixivParserTree(object):
    def __init__(self, tree: Any) -> None:
        super().__init__()
        self.tree = tree

    def xpath(self, path:str):
        eles = self.tree.xpath(path)
        ret = []
        for e in eles:
            ret.append(PixivParserTree(e))
        return ret

    def first(self, path:str) -> Optional[Union['PixivParserTree', str]]:
        lst = self.tree.xpath(path)
        if (len(lst) <= 0 ):
            return None
        else:
            if isinstance(lst[0], str):
                return lst[0]
            else:
                return PixivParserTree(lst[0])

    def text(self):
        lst = self.tree.xpath('./text()')
        if (len(lst) <= 0 ): return None
        else: return lst[0]

    def href(self):
        lst = self.tree.xpath('./@href')
        if (len(lst) <= 0 ): return None
        else: return lst[0]

class PixivParser(object):
    def __init__(self) -> None:
        super().__init__()
        self.parser = etree.HTMLParser()

    def parse(self, html:str):
        tree = etree.fromstring(html, self.parser)
        return PixivParserTree(tree)

class PixivSpider(object):
    def __init__(self, session=None) -> None:
        super().__init__()
        # chrome_options = Options()
        #chrome_options.add_argument("--disable-extensions")
        #chrome_options.add_argument("--disable-gpu")
        #chrome_options.add_argument("--no-sandbox") # linux only
        # chrome_options.add_argument("--headless")
        # chrome_options.headless = True # also works
        self.driver = webdriver.Chrome()
        self.driver.get("https://www.pixiv.net/")
        if session is not None:
            self.driver.add_cookie({
                'name': 'PHPSESSID',
                'value': session,
                'domain': '.pixiv.net',
                'path': '/'
            })
        # self.driver = webdriver.Firefox()
        self.protocol = "https"
        self.domain = "www.pixiv.net"
        
        self.parser = PixivParser()
        self.debug = False
        self.headers = {'Referer': 'https://www.pixiv.net/'}
        self.datapath = "./data/"
        self.waittime = 1

        self.visited_users = set()

        self.xpath = dict()
        self.xpath['username'] = "//h1[contains(@class, 'gwFMaj')]"
        self.xpath['following_number'] = "//span[contains(@class, 'eekErd')]"
        self.xpath['contact'] = "//ul[@class='_2AOtfl9']/li"
        self.xpath['intro'] = "//div[contains(@class, 'cwGEmo')]/text()"
        self.xpath['follow_user'] = "//a[contains(@class, 'bvKIhS')]"
        self.xpath['artwork_thumbnail'] = "//a[contains(@class, 'kdmVAX')]"
        self.xpath['other_links'] = "//a[contains(@class, '_2m8qrc7')]"
        self.xpath['img_url'] = "//a[contains(@class, 'eyBHcd')]/@href"
        self.xpath['label'] = "//span[contains(@class, 'iVbpUG')]/span[1]/a"
        self.xpath['thumbup'] = "//dl[contains(@class, 'fwcFge')]/dd"
        self.xpath['like'] = "//dl[contains(@class, 'doVpzs')]/dd"
        self.xpath['view'] = "//dl[contains(@class, 'fwcFge')]/dd"
        self.xpath['create_time'] = "//div[contains(@class, 'cuUtZw')]"
        

    def start(self) -> None:
        user_list = []
        user_list.append(9041704)
        while len(user_list) > 0:
            now_user = user_list.pop(0)
            if now_user in self.visited_users:
                continue
            else:
                self.visited_users.add(now_user)
            
            self.get_user_all_artworks(now_user)

            following = self.get_following(now_user, 1)
            user_list.extend([f['id'] for f in following])

    def close(self) -> None:
        self.driver.close()

    def get_user_all_artworks(self, user_id: Union[str, int]) -> None:
        user_data = self.get_user(user_id=user_id)
        print(user_data)
        user_path = os.path.join(self.datapath, f'{user_id}/')
        if not os.path.exists(user_path):
            os.mkdir(user_path)

        meta_path = os.path.join(self.datapath, f'{user_id}/user.json')
        if not os.path.exists(meta_path):
            with open(meta_path, 'w', encoding='utf-8') as f:
                user_data['crawl_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(json.dumps(user_data))
        
        artworks_page = 1
        all_artworks = []
        while True:
            artworks = self.get_artworks(user_id=user_id, page=artworks_page)
            artworks_page += 1
            if len(artworks['artworks']) == 0:
                break
            else:
                all_artworks.extend(artworks['artworks'])

        for each_artwork in all_artworks:
            img_path = os.path.join(self.datapath, f'{user_id}/{each_artwork}.jpg')
            meta_path = os.path.join(self.datapath, f'{user_id}/{each_artwork}.json')
            print(img_path)

            if os.path.exists(img_path) and os.path.exists(meta_path):
                continue

            artwork_data = self.get_artwork(each_artwork)
            if artwork_data['url'] is None:
                continue
            res = requests.get(artwork_data['url'], headers=self.headers, verify=False)

            if not os.path.exists(img_path):
                with open(img_path, 'wb') as f:
                    f.write(res.content)
            
            if not os.path.exists(meta_path):
                with open(meta_path, 'w', encoding='utf-8') as f:
                    artwork_data['crawl_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(json.dumps(artwork_data))

    def get_user(self, user_id: Union[str, int]) -> None:
        url = f"{self.protocol}://{self.domain}/users/{user_id}"
        self.driver.get(url)
        time.sleep(self.waittime)

        html = self.driver.page_source
        
        tree = self.parser.parse(html)

        user = dict()
        user['name'] = tree.first(self.xpath['username']).text()
        user['following_number'] = tree.first(self.xpath['following_number']).text()
        contacts = tree.xpath(self.xpath['contact'])
        user['contact'] = [self.convert_jump_url(e.first("./a/@href")) for e in contacts]
        user['intro'] = tree.first(self.xpath['intro'])

        return user

    def get_following(self, user_id: Union[str, int], page: int):
        url =f"{self.protocol}://{self.domain}/users/{user_id}/following?p={page}"
        self.driver.get(url)
        time.sleep(self.waittime)
        html = self.driver.page_source
        
        tree = self.parser.parse(html)

        following_list = []
        users = tree.xpath(self.xpath['follow_user'])
        for user in users:
            following_list.append({
                'name': user.text(),
                'id': int(user.href()[7:].replace(',', ''))
            })

        return following_list

    def get_artworks(self, user_id: Union[str, int], page: int) -> None:
        url = f"{self.protocol}://{self.domain}/users/{user_id}/artworks?p={page}"
        self.driver.get(url)
        time.sleep(self.waittime)
        html = self.driver.page_source
        
        tree = self.parser.parse(html)
        ret = {}

        artworks = []
        thumbnails = tree.xpath(self.xpath['artwork_thumbnail'])
        for artwork in thumbnails:
            artworks.append(int(artwork.href()[10:].replace(',', '')))

        next_links = []
        links = tree.xpath(self.xpath['other_links'])
        for l in links:
            href = l.href()
            if '?p=' not in href:
                continue
            parts = href.split('/')
            next_links.append(int(parts[-1][11:].replace(',', '')))
        ret['artworks'] = artworks
        ret['next_links'] = next_links
        return ret

    def get_artwork(self, artwork_id: Union[str, int]) -> None:
        url = f"{self.protocol}://{self.domain}/artworks/{artwork_id}"
        self.driver.get(url)
        time.sleep(self.waittime)
        html = self.driver.page_source
        
        tree = self.parser.parse(html)
        data = dict()
        data['url'] = tree.first(self.xpath['img_url'])
        data['labels'] = [e.text() for e in tree.xpath(self.xpath['label'])]
        data['thumbup'] = int(tree.first(self.xpath['thumbup']).text().replace(',', ''))
        data['like'] = int(tree.first(self.xpath['like']).text().replace(',', ''))
        data['view'] = int(tree.first(self.xpath['view']).text().replace(',', ''))
        data['create_time'] = tree.first(self.xpath['create_time']).text()
        return data


    def convert_jump_url(self, url: Optional[str]) -> str:
        if url is None:
            return None
        if url.startswith("/jump.php?url="):
            return urllib.parse.unquote(url[14:])
        else:
            return url

    def write_to(self, filepath: str, html: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

if __name__ == "__main__":
    with open("session.txt", 'r', encoding='utf-8') as f:
        session = f.read()
    spider = PixivSpider()
    spider.start(session=session)
    spider.close()
