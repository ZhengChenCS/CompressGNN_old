import re
import os.path as osp
import pandas as pd

log_file = 'benchmark.txt'
with open(log_file) as log:
    content = log.read()
    dataset_name_pattern = re.compile(r'Dataset=/home/KongmingDataset/(.*?)/origin')
    dataset_name = re.findall(dataset_name_pattern, content)
    train_time_pattern = re.compile(r'Train time for \d+ epoch: (\d+\.\d+)s')
    train_time = re.findall(train_time_pattern, content)
    accu_pattern = re.compile(r'Accuracy of Test Samples: (\d+\.\d+)\%')
    accu = re.findall(accu_pattern, content)
    H_pattern = re.compile(r'Param_H=(\d+)')
    H = re.findall(H_pattern, content)
    index_cache_pattern = re.compile(r'index_cache=(\w+)')
    index_cache = re.findall(index_cache_pattern, content)
    reset_step_pattern = re.compile(r'reset_step=(\d+)')
    reset_step = re.findall(reset_step_pattern, content)
    cluster_info_pattern = re.compile(r'[^(warm_up_)]cluster_info: (\[.*?\])')
    cluster_info = re.findall(cluster_info_pattern, content)
    warm_up_cluster_info_pattern = re.compile(r'warm_up_cluster_info: (\[.*?\])')
    warm_up_cluster_info = re.findall(warm_up_cluster_info_pattern, content)
    
df = pd.DataFrame(
    {
        'dataset': dataset_name,
        'H': H, 
        'index_cache': index_cache,
        'reset_step': reset_step,
        'train_time(s)': train_time,
        # 'w_cluster_info': warm_up_cluster_info,
        'cluster_info': cluster_info,
        'accu(%)': accu,
    }
    )

print(df)