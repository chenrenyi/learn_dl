import multiprocessing
import requests
import os
import imghdr

proxies = {'http': '127.0.0.1:1087', 'https': '127.0.0.1:1087'}


def download_image(url, path, i, use_proxy):
    """单个下载方法"""
    name = url.split('/')[-1]
    file = os.path.join(path, name)
    item = requests.get(url, proxies=proxies).content if use_proxy else requests.get(url).content
    os.makedirs(os.path.dirname(file), exist_ok=True)
    print('end download {}'.format(i))

    with open(file, 'wb') as f:
        f.write(item)

    if not imghdr.what(file):
        os.remove(file)


def multi_download(urls, start, end, file, use_proxy=False):
    """多线程下载"""
    pool = multiprocessing.Pool(processes=20)
    for i in range(start, end):
        url = urls[i]
        pool.apply_async(func=download_image, args=(url, file, i, use_proxy))

    pool.close()
    pool.join()


if __name__ == '__main__':
    path = './data/'
    base_urls = {
        # 'dog': 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02111277',
        'cat': 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02123159',
    }

    for image_type in base_urls:
        base_url = base_urls[image_type]
        path = path + image_type
        image_urls = requests.get(base_url, proxies=proxies).text.split('\r\n')
        multi_download(image_urls, 0, 1000, path)
