'''
模型效果测评
    客观评估
'''

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset_processor import DatasetProcessor
from net_model import Model

# 定义设备信息
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载字典和分词器
token = BertTokenizer.from_pretrained(
    r"/Users/avinzhang/git/avin-kit/llm_test/test02/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")


# 将传入的字符串进行编码
def collate_fn(data):
    sents = [i[0] for i in data]
    label = [i[1] for i in data]

    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        # 当句子长度大于 max_length （上限是 model_max_length）时，截断
        truncation=True,
        max_length=512,
        # 一律补0到max_length
        padding="max_length",
        # 可取值为tf,pt,np, 默认为list
        return_tensors="pt",
        # 返回序列长度
        return_length=True
    )

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    label = torch.LongTensor(label)

    return input_ids, attention_mask, token_type_ids, label


# 创建数据集
test_dataset = DatasetProcessor("test")
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=50,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

if __name__ == '__main__':
    acc = 0.0
    total = 0.0

    print(DEVICE)
    model = Model().to(DEVICE)
    # 加载模型训练参数
    model.load_state_dict(torch.load("param/22_bert.pth"))
    # 开启测试模式
    model.eval()

    for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(test_loader):
        # 将数据放在设备上面
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        label = label.to(DEVICE)
        # 前向计算
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = out.argmax(dim=1)
        acc += (out == label).sum().item()
        print(i, (out == label).sum().item())
        total += len(label)
    print(f"test acc: {acc / total}")
