'''
模型训练
    纯训练，无模型验证
'''

from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW

from dataset_processor import DatasetProcessor
from net_model import Model

# 定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义训练轮次（将整个数据集训练完成为一个轮次）
EPOCH = 300000

# 加载字典和分词器
token = BertTokenizer.from_pretrained(
    r"/Users/avinzhang/git/avin-kit/llm_test/test02/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")


# 将传入的字符串进行编码
def collate_fn(data):
    sents = [i[0] for i in data]
    label = [i[1] for i in data]

    # 编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        # 当句子长度大于max_length（上限是model_max_length）时，截断
        truncation=True,
        max_length=512,
        # 一律补0到max_length
        padding="max_length",
        # 可取值为tf,pt,np,默认为list
        return_tensors="pt",
        # 返回序列长度
        return_length=True
    )

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    label = torch.LongTensor(label)
    return input_ids, attention_mask, token_type_ids, label


# 创建数据集
train_dataset = DatasetProcessor("train")
train_loader = DataLoader(
    dataset=train_dataset,
    # 训练批次
    batch_size=80,
    # 打乱数据集
    shuffle=True,
    # 舍弃最后一个批次的数据，防止报错
    drop_last=True,
    # 对加载的数据进行编码
    collate_fn=collate_fn
)

if __name__ == '__main__':
    # 开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    # 定义优化器
    optimizer = AdamW(model.parameters())
    # 定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(train_loader):
            # 将数据加载到DEVICE上面
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            label = label.to(DEVICE)

            # 前向计算 将数据输入模型得到输出
            out = model(input_ids, attention_mask, token_type_ids)
            # 根据损失函数计算损失
            loss = loss_func(out, label)
            # 根据误差优化参数，这三行代码就是ai在学习的过程，反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每隔五个批次，输出训练信息
            if i % 5 == 0:
                out = out.argmax(dim=1)
                # 计算训练精度
                acc = (out == label).sum().item() / len(label)
                time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"time: {time}，epoch: {epoch}, step: {i}, loss: {loss.item()}, acc: {acc}")

        # 每训练完一轮，保存一次参数
        torch.save(model.state_dict(), f"param/{epoch}_bert.pth")
        print(epoch, "参数保存成功！")
