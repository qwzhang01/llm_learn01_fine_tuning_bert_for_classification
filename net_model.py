"""
多层神经网络模型
"""
# 引入 pytorch 库
import torch
# 引入 transformers 的 BertModel类
from transformers import BertModel

# 定义设备，自动选择CUDA显卡或CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 打印设备
print(DEVICE)

# 从本地加载模型
pretrained = BertModel.from_pretrained(
    r"./model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f").to(
    DEVICE)
print(pretrained)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 设计全连接网络，实现二分类任务 模型微调增量部分
        self.fc = torch.nn.Linear(768, 2)
        # 全量微调 增量微调 部分微调

    # 使用模型处理数据，执行前向计算
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 冻结bert模型的参数，让其不参与训练
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 增量模型参与训练
        out = self.fc(out.last_hidden_state[:, 0])
        return out



