# 导入 datasets 的数据加载类
from datasets import load_dataset
# 导入 torch 数据集工具
from torch.utils.data import Dataset

# 定义数据集class
class DatasetProcessor(Dataset):
    # 初始化数据集
    def __init__(self, split):
        # 从磁盘加载数据
        self.dataset = load_dataset(r"./data/lansinuote___chn_senti_corp")
        # 按照数据类型加载数据
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        else:
            print("数据名错误！")

    # 返回数据集的长度
    def __len__(self):
        return len(self.dataset)

    # 对每条数据单独做处理
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]

        return text, label

# 测试数据集
if __name__ == '__main__':
    dataset = DatasetProcessor("train")
    for data in dataset:
        print(data)
