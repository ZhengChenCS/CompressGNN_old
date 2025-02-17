# root_dir=/home/chenzheng/Kongming/CompressGNNDataset/dataset
root_dir=/mnt/disk1/GNNDataset

dataset=(Cora CiteSeer PubMed Reddit AmazonProducts)
num_features=(1433 3703 500 602 200)
num_classes=(7 6 3 41 102) 
deltaH=(10 6 10 4 5)

#origin SGC

# for i in {0..4}
# do
#     data=${root_dir}/${dataset[i]}/origin/data_csr.pt
#     python sgc.py \
#     --data=${data} \
#     --num_features=${num_features[i]} \
#     --num_classes=${num_classes[i]} \
#     --epochs=100000 \
#     --device=gpu
#     # break
# done

for i in {4..4}
do
    data=${root_dir}/${dataset[i]}/origin/data_csr.pt
    python compressgnn_sgc.py \
    --data=${data} \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100000 \
    --device=gpu \
    --deltaH=${deltaH[i]}
done

# for i in {0..4}
# do
#     data=${root_dir}/${dataset[i]}/origin/data_csr.pt
#     python gcn.py \
#     --data=${data} \
#     --num_features=${num_features[i]} \
#     --num_classes=${num_classes[i]} \
#     --epochs=100000 \
#     --device=gpu
#     # break
# done

