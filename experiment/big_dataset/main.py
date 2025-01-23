import torch
import numpy as np
import dgl
# from ogb.nodeproppred import DglNodePropPredDataset, PygNodePropPredDataset



if __name__ == "__main__":
    path = "/mnt/disk1/GNNDataset/OGB/dataset/ogbn_papers100M"
    
    datapath = f"{path}/raw/data.npz"
    data = np.load(datapath)
    '''
    node_feat
    edge_index
    num_nodes
    num_edges
    node_year
    '''
    # for key in data.files:
    #     print(key)
    # print(data)
    print(data['node_feat'].shape)
