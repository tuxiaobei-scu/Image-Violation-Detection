import easyocr
reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory

# 读取 dataset 目录下的图片
import os
types = ['normal', 'porn', 'sensitive']
for type in types:
    for img in os.listdir(f'dataset/train/{type}'):
        if img[-3:]!= 'jpg':
            continue
        result = reader.readtext(f'dataset/train/{type}/{img}', detail = 0)
        ret =' '.join(result)
        print(img, ret)
        open(f'dataset/train/{type}/{img[:-4]}.txt', 'w', encoding='utf-8').write(ret)