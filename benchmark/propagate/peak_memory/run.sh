root_dir=/home/KongmingDataset
ncu=/home/chenzheng/.conda/envs/local_pyg/nsight-compute/ncu

dataset=(cnr-2000 web-BerkStan eu-2005 in-2004 uk-2007-05@1000000 hollywood-2009 Reddit AmazonProducts)

for i in {0..5}
do
    compress_coo_data=${root_dir}/${dataset[i]}/compress/data_coo_3.pt
    compress_csr_data=${root_dir}/${dataset[i]}/compress/data_csr_3.pt
    coo_data=${root_dir}/${dataset[i]}/origin/data_coo.pt
    csr_data=${root_dir}/${dataset[i]}/origin/data_csr.pt
    echo ${dataset[i]}
    echo "----------Kongming GAS-----------"
    python kongming_gas.py \
    --data=${compress_coo_data}

    echo "----------Kongming SPMM-----------"
    python kongming_spmm.py \
    --data=${compress_csr_data}

    echo "----------PyG GAS-----------"
    python pyg_gas.py \
    --data=${coo_data}

    echo "----------PyG SPMM-----------"
    python pyg_spmm.py \
    --data=${csr_data}

    echo "----------DGL SPMM-----------"
    python dgl_spmm.py \
    --data=${csr_data} 
done

for i in {6..7}
do
    compress_coo_data=${root_dir}/${dataset[i]}/compress/data_coo_3.pt
    compress_csr_data=${root_dir}/${dataset[i]}/compress/data_csr_3.pt
    coo_data=${root_dir}/${dataset[i]}/origin/data_coo.pt
    csr_data=${root_dir}/${dataset[i]}/origin/data_csr.pt
    echo ${dataset[i]}

    echo "----------Kongming SPMM-----------"
    python kongming_spmm.py \
    --data=${compress_csr_data}


    echo "----------PyG SPMM-----------"
    python pyg_spmm.py \
    --data=${csr_data}

    echo "----------DGL SPMM-----------"
    python dgl_spmm.py \
    --data=${csr_data} 
done