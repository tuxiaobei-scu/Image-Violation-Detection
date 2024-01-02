import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, BertTokenizer, BertModel, AdamW
from torchvision import datasets, transforms, models
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import os
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 加载数据
class CustomDataset(Dataset):
    def __init__(self, data_folder, tokenizer, max_len):
        self.data_folder = data_folder
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        class_list = ['normal', 'porn','sensitive']
        for t in class_list:
            path = f'{data_folder}/{t}'
            for file in os.listdir(path):
                if file.endswith('.txt'):
                    file_name = os.path.join(path, file[:-4])
                    self.data.append([file_name, class_list.index(t)])

            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = open(self.data[index][0] + '.txt', 'r', encoding='utf-8').read()
        img = Image.open(self.data[index][0] + '.jpg')
        img = transform(img)
        label = self.data[index][1]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'img': img,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

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
    
# 加载模型和tokenizer
model_name = 'transformers_models/ernie-3.0-mini-zh'  # 可根据实际需要更改
tokenizer = BertTokenizer.from_pretrained(model_name)
model = MyModel(num_labels=3)
model.load_state_dict(torch.load('final_model.pth'))

# 定义超参数
max_len = 256
batch_size = 64
learning_rate = 1e-5
epochs = 25
save_dir = 'final_model_v2'

# 创建数据加载器
train_dataset = CustomDataset('dataset/train', tokenizer, max_len)
val_dataset = CustomDataset('dataset/val', tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义优化器和损失函数
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
scaler = GradScaler()
for epoch in range(epochs):
    model.train()
    total_loss = 0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
        for batch in train_loader:
            img = batch['img'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            
            with autocast():
                outputs = model(img, input_ids, attention_mask=attention_mask, infer=False)
                logits = outputs

                loss = criterion(logits, labels)
                total_loss += loss.item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.update(1)

    average_loss = total_loss / len(train_loader)
    print(f'Training Loss: {average_loss}')

    # 在验证集上评估模型
    model.eval()
    val_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            img = batch['img'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(img, input_ids, attention_mask=attention_mask)
            logits = outputs

            val_loss += criterion(logits, labels).item()
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_val_loss = val_loss / len(val_loader)
    accuracy = torch.sum(torch.tensor(all_predictions) == torch.tensor(all_labels)).item() / len(val_dataset)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

    print(f'Validation Loss: {average_val_loss}, Accuracy: {accuracy}')
    print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')

    open('final_train_v2.log', 'a').write(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}\n')

    model_path = os.path.join(save_dir, f'bert_epoch{epoch+1}_best.pth')
    torch.save(model.state_dict(), model_path)
