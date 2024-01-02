from transformers import BertTokenizer, AdamW, BertTokenizer, BertModel, AdamW
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import easyocr
import numpy as np
from PIL import Image
import onnx
import os


reader = easyocr.Reader(['ch_sim','en']) 
tokenizer = BertTokenizer.from_pretrained('transformers_models/ernie-3.0-mini-zh')

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class MyModel(nn.Module):
    def __init__(self, num_labels):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained('transformers_models/ernie-3.0-mini-zh-fine-tuned')
        self.model = models.inception_v3()
        img_features_dim = 512
        self.model.fc = nn.Linear(self.model.fc.in_features, img_features_dim)
        self.fc1 = nn.Linear(img_features_dim + self.bert.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_labels)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.orthogonal_(self.model.fc.weight)
        nn.init.zeros_(self.model.fc.bias)

    def forward(self, img, input_ids, attention_mask, infer=True):
        if infer:
            img_output = self.model(img)
        else:
            img_output = self.model(img).logits
        img_output = self.relu(img_output)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        fc1_input = torch.cat((pooled_output, img_output), dim=1)
        fc1_output = self.relu(self.fc1(fc1_input))
        logits = self.fc2(fc1_output)
        return logits
    
model = MyModel(num_labels=3)
model.load_state_dict(torch.load('final_model_v2.pth'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_input_ids(ocr_result):
    encoding = tokenizer(ocr_result, max_length=256, truncation=True, padding='max_length', add_special_tokens=True)
    return encoding['input_ids'], encoding['attention_mask']

# 读取 dataset/val 目录下的图片
types = ['normal', 'porn', 'sensitive'] 
for i in range(len(types)):
    path = 'dataset/val/' + types[i] + '/'
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            continue
        filename = file.split('.')[0]
        pic = path + filename + '.jpg'
        img = Image.open(pic)
        img = transform(img).unsqueeze(0)
        ocr_result = open(path + filename + '.txt', 'r', encoding='utf-8').read()
        text_input_ids, text_attention_mask = get_input_ids(ocr_result)
        text_input_ids = torch.tensor(text_input_ids).unsqueeze(0)
        text_attention_mask = torch.tensor(text_attention_mask).unsqueeze(0)
        with torch.no_grad():
            img = img.to(device)
            text_input_ids = text_input_ids.to(device)
            text_attention_mask = text_attention_mask.to(device)
            logits = model(img, text_input_ids, text_attention_mask)
            # softmax
            probs = nn.functional.softmax(logits, dim=1)
            open('val_result/' + types[i] + '.csv', 'a', encoding='utf-8').write(filename + ',' + str(probs.tolist()[0][0]) + ',' + str(probs.tolist()[0][1]) + ',' + str(probs.tolist()[0][2]) + '\n')
            