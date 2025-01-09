import torch
import dgl
from dgl.dataloading import MultiLayerNeighborSampler, DataLoader
import time
import sys
sys.path.append("../loader")
from origin_data import Data

def main():

    data_path = "/mnt/disk1/KongmingDataset/cnr-2000/origin/data_csr.pt"
    data = torch.load(data_path)
    adj = data.edge_index
    src, dst, _ = adj.coo()
    num_nodes = adj.size(0)
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g = g.formats(['csc'])

    # 定义种子节点（如训练节点）
    train_nids = torch.arange(num_nodes)

    # 定义采样器
    num_neighbors = [25, 10, 10]
    sampler = MultiLayerNeighborSampler(num_neighbors)

    # 创建数据加载器
    batch_size = 1024
    num_workers = 1  # 根据机器配置调整
    dataloader = DataLoader(
        g,
        train_nids,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )

    start = time.perf_counter()
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        # 模拟训练步骤
        print("hello")
        pass
    end = time.perf_counter()
    print("Total sampling time: {:.2f}s".format(end - start))

if __name__ == '__main__':
    main()