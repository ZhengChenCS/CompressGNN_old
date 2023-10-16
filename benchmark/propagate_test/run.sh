root_dir=/home/KongmingDataset
ncu=/home/chenzheng/.conda/envs/local_pyg/nsight-compute/ncu

dataset=(cnr-2000 web-BerkStan eu-2005 in-2004 uk-2007-05@1000000 hollywood-2009 Reddit AmazonProducts)

for i in {0..0}
do
    echo ${dataset[i]}
    # echo "DGL Sparse:"
    # python dgl_sparse.py ${dataset[i]}
    # echo "PyG Sparse:"
    # python pyg_sparse.py ${dataset[i]}
    # echo "Kongming SPMM:"
    # python kongming_spmm.py ${dataset[i]}
    python check.py ${dataset[i]}
done

# for i in {0..0}
# do
#     echo ${dataset[i]}
#     echo "PyG Scatter:"
#     python pyg_scatter.py ${dataset[i]}
#     break
# done