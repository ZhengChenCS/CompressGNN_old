import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import sys
sys.path.append("../layer/")
sys.path.append("../loader/")
sys.path.append("../model")
from origin_data import Data
import numpy as np
import compressgnn_offline


def reorder_data(edge_index):
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    print("start reorder")
    try:
        src, dst, order = compressgnn_offline.reorder(src, dst, 5)
        src = np.array(src)
        dst = np.array(dst)
        order = np.array(order)
        print("reorder done")
        return src, dst, order
    except Exception as e:
        print(e)
        return None, None, None

def check_reorder(vlist, elist, rvlist, relist, order):
    for i in range(len(vlist)-1):
        ori_degree = vlist[i+1] - vlist[i]
        re_degree = rvlist[order[i]+1] - rvlist[order[i]]
        
        neighbors = elist[vlist[i]:vlist[i+1]]
        re_neighbors = relist[rvlist[order[i]]:rvlist[order[i]+1]]
        neighbors = order[neighbors]
        neighbors = np.sort(neighbors)
        if not np.array_equal(neighbors, re_neighbors):
            print(i)
            print(neighbors)
            print(re_neighbors)
            print("=================")
            break


# src = [0, 1, 1, 2, 2, 2]
# dst = [1, 0, 2, 0, 1, 3]

'''
Transform:
[0, 0, 0, 1, 1, 2]
[1, 2, 3, 0, 2, 1]
origin -> transform
0 -> 2
1 -> 1
2 -> 0
3 -> 3
'''


'''
[0, 0, 1, 2, 2, 2]
[1, 2, 0, 0, 1, 3]
0 -> 1
1 -> 0
2 -> 2
3 -> 3
'''

# origin[first_order] = transform
# transform[order] = reorder

# origin[order[first_order]] = reorder

# src, dst, order = compressgnn_offline.reorder(src, dst, 5)
# print(src)
# print(dst)
# print(order)


edge_index = np.load("/mnt/disk1/GNNDataset/Reddit/origin/edge.npy")
v_cnt = edge_index[0].max() + 1
vlist, elist = compressgnn_offline.coo2csr(edge_index[0], edge_index[1], v_cnt)

# src, dst, order = reorder_data(edge_index)
# src = np.array(src)
# dst = np.array(dst)
# order = np.array(order)
# np.save("order.npy", order)

# rvlist, relist = compressgnn_offline.coo2csr(src, dst, v_cnt)
# np.save("rvlist.npy", rvlist)
# np.save("relist.npy", relist)

order = np.load("order.npy")
rvlist = np.load("rvlist.npy")
relist = np.load("relist.npy")


check_reorder(vlist, elist, rvlist, relist, order)











 


