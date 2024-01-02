

import requests
import json
import random
import time
import threading
import json
import hashlib
import base64
from PIL import Image, ImageSequence
def md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

conf = json.load(open('.conf.json'))

groups = conf['groups']
api_url = conf['api_url']
verifyKey = conf['verifyKey']
bot_qq = conf['bot_qq']
pic_check_api = conf['pic_check_api']
threshold = conf['threshold']

qq_ua_header = {
    'User-Agent': 'Mozilla/5.0 (Linux; Android 10; Redmi K30 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Mobile Safari/537.36 QQ/8.4.17.612 V1_AND_SQ_8.4.17_0_TIM_D TIM/3.0.0.3080 QQTheme/1000',
    'Referer': 'http://gchat.qpic.cn',
}

def request(name, data):
    return requests.post(url=f'{api_url}/' + name, data=json.dumps(data, ensure_ascii=False).encode('utf-8')).json()


sessionKey = request('verify', {"verifyKey": verifyKey})['session']

def getsessionKey(sessionKey):
    if (request('bind', {"sessionKey": sessionKey, "qq": bot_qq})['code'] == 0):
        return sessionKey
    sessionKey = request('verify', {"verifyKey": verifyKey})['session']
    request('bind', {"sessionKey": sessionKey, "qq": bot_qq})
    return sessionKey


def sendGroupMessage(group, msg):
    global sessionKey
    sessionKey = getsessionKey(sessionKey)
    print(request('sendGroupMessage', {
        'sessionKey': sessionKey, 'group': group, 'messageChain': msg}))


def recallGroupMessage(msgid, groupid):
    global sessionKey
    sessionKey = getsessionKey(sessionKey)
    print(request('recall', {
        'sessionKey': sessionKey, 'messageId': msgid, 'target': groupid}))


def getMessage(count=100):
    global sessionKey
    sessionKey = getsessionKey(sessionKey)
    url = f'{api_url}/fetchLatestMessage?sessionKey=%s&count=%d' % (
        sessionKey, count)
    return requests.get(url).json()['data']


def recall(msgid, groupid, msg, reason=' 你的图片涉嫌违规，请注意言行！'):
    recallGroupMessage(msgid, groupid)
    ret = []
    ret.append({'type': 'At', 'target': int(msg['sender']['id']), 'display': '@' + msg['sender']['memberName']})
    ret.append({'type': 'Plain', 'text': reason})
    sendGroupMessage(groupid, ret)

def pre_pic(pic):
    img = Image.open(pic)
    img = img.convert('RGB')
    img.save(pic, 'JPEG')
        

def calc1(msg):
    sid = msg['messageChain'][0]['id']
    s = ''
    imgs = []
    for p in msg['messageChain']:
        if p['type'] == 'Image':
            file_name = f'img/{md5(p["url"])}.jpg'
            with open(file_name, 'wb') as f:
                f.write(requests.get(p['url'], headers=qq_ua_header).content)
            pre_pic(file_name)
            imgs.append(file_name)
    for img in imgs:
        base64_data = base64.b64encode(open(img, 'rb').read())
        r = requests.post(pic_check_api, data={'base64': base64_data})
        data = r.json()
        data['sender'] = msg['sender']
        open('log.json', 'a', encoding='utf-8').write(json.dumps(data, ensure_ascii=False) + '\n')
        if data['code'] != 0:
            continue
        data = data['data']
        for res in data['imgclass_result']:
            if res['label'] != 'normal' and res['prob'] > threshold:
                recall(sid, msg['sender']['group']['id'], msg, ' 你的图片涉嫌违规，违规置信度 {:.2f} %，请注意言行！'.format(res['prob'] * 100))
                return

        if data['sensitive_detect']['category'] != 'None':
            recall(sid, msg['sender']['group']['id'], msg , ' 你的图片内容所含敏感词，请注意言行！')
            return

        for p in data['malicious_detect']:
            if p['code'] != 200:
                recall(sid, msg['sender']['group']['id'], msg, ' 你的图片含有违规二维码，请注意言行！')
                return

isopen = True


def calcmessage():
    message = getMessage()
    global isopen
    for msg in message:
        if msg['type'] != 'GroupMessage' or msg['sender']['group']['id'] not in groups:
            continue
        if msg['sender']['permission'] != 'MEMBER':
            s = ''
            for p in msg['messageChain']:
                if p['type'] == 'Plain':
                    s += p['text']
            if '开启图片审核' in s:
                isopen = True
                sendGroupMessage(msg['sender']['group']['id'], [
                                 {'type': 'Plain', 'text': '图片审核已开启'}])
            elif '关闭图片审核' in s:
                isopen = False
                sendGroupMessage(msg['sender']['group']['id'], [
                                 {'type': 'Plain', 'text': '图片审核已关闭'}])
        if not isopen:
            continue
        threading._start_new_thread(calc1, (msg,))
        # calc2(msg['sender'])

print('开始运行')
while True:
    calcmessage()
    time.sleep(0.5)
