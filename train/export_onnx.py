from transformers import BertTokenizer, AdamW, BertTokenizer, BertModel, AdamW
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import easyocr
import numpy as np
from PIL import Image
import onnx


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

img_input = torch.randn(1, 3, 299, 299)  # 299x299 是 Inception V3 的输入大小
input_ids = torch.LongTensor([[1] * 256])
attention_mask = torch.ones_like(input_ids)

# 导出模型为ONNX格式
onnx_export_path = "final_model_v2.onnx"
torch.onnx.export(
    model,
    (img_input, input_ids, attention_mask),
    onnx_export_path,
    verbose=True,
    input_names=["img", "input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "img": {0: "batch_size"},
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

onnx_model = onnx.load(onnx_export_path)
print(onnx.checker.check_model(onnx_model))