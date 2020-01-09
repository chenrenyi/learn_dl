import requests
from pytorch.spider import spider
import shutil

base_url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=123&cl=2' \
           '&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&hd=&latest=&copyright=&word={}&s=&se=&tab=&width=&height=&face' \
           '=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn={}&rn={}&gsm=&1578552699562= '

page_size = 30
image_start_num = 1
max_num_count = 600
val_proportion = 0.3
words = {'man': '男', 'woman': '女'}

shutil.rmtree('../data/dog_cats/train')
shutil.rmtree('../data/dog_cats/val')

for word_key in words:
    word = words[word_key]
    image_urls = []
    # get image urls from baidu image
    for i in range(max_num_count // page_size + 1):
        current_start_num = i * page_size
        url = base_url.format(word, current_start_num, page_size)
        try:
            response = requests.get(url).json()
        except:
            continue

        for item in response['data']:
            if 'thumbURL' in item:
                image_urls.append(item['thumbURL'])

    # split image urls into train sub and val sub and download it
    image_total_num = len(image_urls)
    val_image_num = int(image_total_num * val_proportion)
    spider.multi_download(image_urls, 0, val_image_num - 1, '../data/dog_cats/val/' + word_key)
    spider.multi_download(image_urls, val_image_num, image_total_num - 1, '../data/dog_cats/train/' + word_key)

    print('download {} images success! total: {}, train: {}, val: {}'.format(word,
                                                                             image_total_num,
                                                                             image_total_num - val_image_num,
                                                                             val_image_num))
