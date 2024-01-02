import json
import onnx
import onnxruntime
import torch
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image
import easyocr
import numpy as np
from flask import Flask, request, jsonify
import random
import base64
import requests
import hashlib
import os
import urllib
import hnswlib


import cv2
from pyzbar.pyzbar import decode
import re

def extract_links_from_qr_code(image_path):
    img = cv2.imread(image_path)

    # 解码二维码
    decoded_objects = decode(img)

    # 提取链接
    links = []
    for obj in decoded_objects:
        if obj.type == 'QRCODE' and obj.data:
            link = obj.data.decode('utf-8')
            links.append(link)

    # 正则匹配合法链接
    pattern = r"(https?://\S+)"
    valid_links = []
    for link in links:
        matches = re.findall(pattern, link)
        if matches:
            valid_links.extend(matches)

    return valid_links

def md5(str):
    hl = hashlib.md5()
    hl.update(str.encode())
    return hl.hexdigest()


# 加载 ONNX 模型

tokenizer = BertTokenizer.from_pretrained('ernie-3.0-mini-zh')

onnx_model_path = "model_main.onnx"
onnx_model = onnx.load(onnx_model_path)
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# 创建 ONNX 运行时会话

if 'easyocr' in os.environ and os.environ['easyocr'] == 'true':
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, model_storage_directory='easyocr/model',
                            user_network_directory='easyocr/user_network')

# 准备输入数据

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def get_input_ids(ocr_result):
    encoding = tokenizer(ocr_result, max_length=256, truncation=True, padding='max_length', add_special_tokens=True)
    return encoding['input_ids'], encoding['attention_mask']


def get_access_token(client_id, client_secret):
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
    return str(requests.post(url, params=params).json().get("access_token"))


def ocr(img):
    if 'ocr_type' in os.environ:
        ocr_type = os.environ['ocr_type']
        if ocr_type == 'easyocr':
            ocr_result = reader.readtext(img, detail=0)
            ocr_result = ' '.join(ocr_result)
        elif ocr_type == 'baidu':
            ocr_client_id = os.environ['ocr_client_id']
            ocr_client_secret = os.environ['ocr_client_secret']
            with open(img, "rb") as f:
                content = base64.b64encode(f.read()).decode("utf8")
            content = urllib.parse.quote_plus(content)
            url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token=" + get_access_token(
                ocr_client_id, ocr_client_secret)
            payload = f'image={content}&detect_direction=true'
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload)
            ocr_result = ' '.join([item['words'] for item in response.json()['words_result']])
        else:
            ocr_result = ''
    else:
        ocr_result = ''
    return ocr_result

def detect_malicious_website(url):
    params = {
        'apiKey': os.environ['mal_api_key'], #71dbd0f9a2a51af019273c9a4075408e
        'url': url
    }
    response = requests.get('https://api.ooomn.com/api/qqsafe', params=params)
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print('请求失败:', response.status_code)

def detect_sensitive_word(content):
    if 'sensitive_detect_type' in os.environ:
        type = os.environ['sensitive_detect_type']
        if type == 'remote':
            url = "https://api.wordscheck.com/check"
            data = json.dumps(
                {'key': os.environ['sen_api_key'], 'content': content})  # aQprzN0lqqxQeW67Qg6qHBUnfJ7W4C6N
            headers = {'Content-Type': 'application/json'}

            response = requests.post(url, data=data, headers=headers)
            data = json.loads(response.text)
            if data['word_list'] == []:
                return ["None", "None"]
            return [data['word_list'][0]['keyword'], data['word_list'][0]['category']]

        elif type == 'local':
            from checkSensitive.src.sensitiveApi import DFAFilter
            model = DFAFilter()
            result = model.detect(content)
            return [result['ifContainSensitiveWord'], result['sensitiveWordList']]
        else:
            return ["None", "None"]
    else:
        return ["None", "None"]
    
transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

search_onnx_model_path = "model_search.onnx"
search_onnx_model = onnx.load(search_onnx_model_path)
search_ort_session = onnxruntime.InferenceSession(search_onnx_model_path)
def get_img_feature(img_path):
    img = Image.open(img_path)
    if img.mode!= 'RGB':
        img = img.convert('RGB')
    
    img = transform_resnet(img).unsqueeze(0)
    ort_inputs = {'img': img.numpy()}
    ort_outputs = search_ort_session.run(None, ort_inputs)
    img_feature = ort_outputs[0]
    img_feature = np.squeeze(img_feature)
    return img_feature

search_dim = 512
pic_database = hnswlib.Index(space = 'cosine', dim = search_dim)
pic_name = {}
white_cnt = 0
black_cnt = 0
def load_img(directory):
    global white_cnt, black_cnt
    data = []
    ids = []
    id = 0
    if not os.path.exists(directory):
        print('img_data not exist')
        return
    if os.path.exists(f'{directory}/white'):
        for filename in os.listdir(f'{directory}/white'):
            img_path = os.path.join(f'{directory}/white', filename)
            feature_vector = get_img_feature(img_path)
            data.append(feature_vector)
            ids.append(id * 2)
            pic_name[id * 2] = filename
            id += 1
            white_cnt += 1

    id = 0
    if os.path.exists(f'{directory}/black'):
        for filename in os.listdir(f'{directory}/black'):
            img_path = os.path.join(f'{directory}/black', filename)
            feature_vector = get_img_feature(img_path)
            data.append(feature_vector)
            ids.append(id * 2 + 1)
            pic_name[id * 2 + 1] = filename
            id += 1
            black_cnt += 1
    if len(data) > 0:
        pic_database.init_index(max_elements=len(data), ef_construction=200, M=16)
        pic_database.add_items(data, ids)
        pic_database.set_ef(50)

def search_img(img_path):
    global white_cnt, black_cnt
    feature_vector = get_img_feature(img_path)
    white_result = []
    black_result = []
    if 'knn_query_k' in os.environ:
        knn_query_k = int(os.environ['knn_query_k'])
    else:
        knn_query_k = 1
    if white_cnt > 0:
        filter_function = lambda idx: idx%2 == 0
        white_ids, white_dists = pic_database.knn_query([feature_vector], k=min(white_cnt, knn_query_k), num_threads=1, filter=filter_function)
        white_ids = white_ids[0]
        white_dists = white_dists[0]
        for i in range(len(white_ids)):
            white_result.append({'filename': pic_name[white_ids[i]], 'sim': 1 - white_dists[i]})
            
    if black_cnt > 0:
        filter_function = lambda idx: idx%2 == 1
        black_ids, black_dists = pic_database.knn_query([feature_vector], k=min(black_cnt, knn_query_k), num_threads=1, filter=filter_function)
        black_ids = black_ids[0]
        black_dists = black_dists[0]
        for i in range(len(black_ids)):
            black_result.append({'filename': pic_name[black_ids[i]], 'sim': 1 - black_dists[i]})
    return white_result, black_result

def get_model_result(pic):
    white_result, black_result = search_img(pic)
    ocr_result = ocr(pic)
    img = Image.open(pic)
    if img.mode!= 'RGB':
        img = img.convert('RGB')
    img = transform(img).unsqueeze(0)

    text_input_ids, text_attention_mask = get_input_ids(ocr_result)
    text_input_ids = torch.tensor(text_input_ids).unsqueeze(0)
    text_attention_mask = torch.tensor(text_attention_mask).unsqueeze(0)

    # 在 ONNX 模型中运行推理
    ort_inputs = {'img': img.numpy(), 'input_ids': text_input_ids.numpy(),
                  'attention_mask': text_attention_mask.numpy()}

    ort_outputs = ort_session.run(None, ort_inputs)

    # 对结果进行 softmax 处理
    raw_scores = ort_outputs[0]

    # 应用 softmax
    probs = np.exp(raw_scores) / np.exp(raw_scores).sum(axis=1, keepdims=True)

    # 输出每个标签的预测概率
    labels = ['normal', 'porn', 'sensitive']  # 你的类别标签
    ret = {}
    for label, prob in zip(labels, probs[0]):
        ret[label] = float(prob)

    # 检测铭感词
    sensitiveResult = detect_sensitive_word(ocr_result)

    # 检测恶意链接
    malicious_result = []
    links = extract_links_from_qr_code(pic)
    for link in links:
         malicious_result.append(detect_malicious_website(link))

    final_result = {
        "img": pic,
        "white_result": white_result,
        "black_result": black_result,
        "imgclassresult":[
            {
                "label": 'normal',
                "prob": ret['normal']
            },
            {
                "label": 'porn',
                "prob": ret['porn']
            },
            {
                "label": 'sensitive',
                "prob": ret['sensitive']
            }
        ],
        "sensitive_detect":{
            "keyword": sensitiveResult[0],
            "category": sensitiveResult[1]
        },
        "malicious_detect":malicious_result
    }

    return final_result


app = Flask(__name__)


@app.route('/check_img', methods=['POST','GET'])
def predict():
    if 'base64' in request.form:
        base64_data = request.form['base64']
        tmp_file_name = 'tmp/' + md5(base64_data)
        with open(tmp_file_name, 'wb') as f:
            f.write(base64.b64decode(base64_data))
    elif 'url' in request.form:
        url = request.form['url']
        tmp_file_name = 'tmp/' + md5(url)
        img = Image.open(requests.get(url, stream=True).raw)
        img.save(tmp_file_name)
    else:
        return jsonify({'code': 1, 'msg': '参数错误'})
    ret = get_model_result(tmp_file_name)
    os.remove(tmp_file_name)
    return jsonify({'code': 0, 'msg': 'success', 'data': ret})

load_img('img_data')
print('load img finish')
print('white_cnt:', white_cnt)
print('black_cnt:', black_cnt)
app.run(host='0.0.0.0', port=5500, debug=False)