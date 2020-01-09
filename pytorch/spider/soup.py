import requests
from bs4 import BeautifulSoup

url = 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=index&fr=&hs=0' \
      '&xthttps=111111&sf=1&fmq=&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word' \
      '=123&oq=123&rsp=-1'
response = requests.get(url)
response.encoding = "utf-8"
html_doc = response.text

soup = BeautifulSoup(html_doc, 'lxml')
titles = soup.select('.imgbox')

print(html_doc)
print(titles)
for title in titles:
    print(title.get('data-imgurl'))
