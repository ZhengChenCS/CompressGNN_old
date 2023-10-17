import os
import sys
sys.path.append("../loader/")
from CompressgnnData import CompressgnnData
import getopt
import torch
import numpy as np

def main():
    data_dir = sys.argv[1]
    graph_dir = sys.argv[1] + "/origin"
    save_dir = sys.argv[2]
    graph_type = sys.argv[3]
    train_mask_path = data_dir + "/train_mask.npy"
    vaild_mask_path = data_dir + "/val_mask.npy"
    test_mask_path = data_dir + "/test_mask.npy"

    x = np.load(data_dir + "/features.npy")
    y = np.load(data_dir + "/labels.npy")
    if y.ndim == 2:
        y = [np.argmax(row_y) for row_y in y]
    y = np.array(y)
    vertex_cnt = y.size
    if not os.path.exists(train_mask_path):
        train_mask = np.ones(vertex_cnt, dtype=bool)
        vaild_mask = np.ones(vertex_cnt, dtype=bool)
        test_mask = np.ones(vertex_cnt, dtype=bool)
    else:
        train_mask = np.load(train_mask_path)
        vaild_mask = np.load(vaild_mask_path)
        test_mask = np.load(test_mask_path)
    if graph_type == "coo":
        edge_index = np.load(graph_dir + "/edge.npy")
        data = CompressgnnData(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            vaild_mask=vaild_mask,
            test_mask=test_mask,
            graph_type="coo",
            max_depth=3,
            threshold=16,
            min_edge=1000000,
            add_self_loop=True)
    elif graph_type == "csr":
        vlist = np.load(graph_dir + "/csr_vlist.npy")
        elist = np.load(graph_dir + "/csr_elist.npy")
        data = CompressgnnData(
            x=x,
            y=y,
            edge_index=(
                vlist,
                elist),
            train_mask=train_mask,
            vaild_mask=vaild_mask,
            test_mask=test_mask,
            graph_type="csr",
            normalize=True,
            max_depth=3,
            min_edge=1000000,
            threshold=16)
    print(data)
    # torch.save(data, save_dir + "/data_" + graph_type + "_default.pt")
    torch.save(data, save_dir + "/data_" + graph_type + "_" + str(data.step) + ".pt")


if __name__ == '__main__':
    main()