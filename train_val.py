'''
模型训练
    验证模型
'''
# 导入 pytorch 库
import torch
# 导入 pytorch 的 DataLoader 类
from torch.utils.data import DataLoader
# 导入 transformers 的 BertTokenizer，AdamW
# AdamW AdamW 是一种优化算法，全称是 Adam with Weight Decay（带权重衰减的 Adam）。它是对经典 Adam（Adaptive Moment Estimation，自适应矩估计）优化器的改进，旨在更好地处理权重衰减（weight decay），提高模型的泛化能力
from transformers import BertTokenizer, AdamW

# 导入自定义的数据集和神经网络模型
from dataset_processor import DatasetProcessor
from net_model import Model

# 定义设备信息
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义训练的轮次（将整个数据集训练完一次为一轮）
EPOCH = 30000

# 加载字典和分词器
token = BertTokenizer.from_pretrained(
    r"./model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

# 将传入的字符串进行编码
def collate_fn(data):
    sents = [i[0] for i in data]
    label = [i[1] for i in data]

    # 文本编码
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

    # 返回编码后的Token
    return input_ids, attention_mask, token_type_ids, label

# 创建训练数据集
train_dataset = DatasetProcessor("train")
train_loader = DataLoader(
    dataset=train_dataset,
    # 真实训练中，批次越大，训练速度和训练效果越好，当然显卡要求越高
    batch_size=50,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)
# 创建验证数据集
val_dataset = DatasetProcessor("validation")
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=50,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

# 执行训练main函数
if __name__ == '__main__':
    print(DEVICE)
    # 将模型加载到设备上
    model = Model().to(DEVICE)
    # 定义优化器
    optimizer = AdamW(model.parameters())
    # 定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    best_val_ass = 0.0

    for epoch in range(EPOCH):
        for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(train_loader):
            # 将数据放到 DEVICE 上面
            input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE), attention_mask.to(
                DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)
            # 前向计算，将数据输入模型得到输出
            out = model(input_ids, attention_mask, token_type_ids)
            # 根据输出计算损失
            loss = loss_func(out, label)
            # 根据误差优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                out = out.argmax(dim=1)
                acc = (out == label).sum().item() / len(label)
                print(f"epoch: {epoch}, i:{i}, loss: {loss.item()}, acc: {acc}")

        # 验证模型 判断模型是否会过拟合
        # 设置为评估模型
        model.eval()
        # 不需要参与训练
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(val_loader):
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                token_type_ids = token_type_ids.to(DEVICE)
                label = label.to(DEVICE)

                # 前向计算 将数据输入到模型得到输出
                out = model(input_ids, attention_mask, token_type_ids)
                # 根据输出计算损失
                val_loss += loss_func(out, label)
                # 根据输出计算验证精度
                out = out.argmax(dim=1)
                val_acc += (out == label).sum().item();
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            print(f"验证集：loss:{val_loss}, acc:{val_acc}")

            # 根据验证准确率保存最优参数
            if val_acc > best_val_ass:
                best_val_ass = val_acc
                torch.save(model.state_dict(), f"param/best_bert.pth")
                print(f"EPOCH：{epoch}：保存最优参数：acc{best_val_ass}")

        # 保存最后一轮参数
        torch.save(model.state_dict(), f"param/last_bert.pth")
        print(f"EPOCH：{epoch}：保存最后一轮参数成功！")
