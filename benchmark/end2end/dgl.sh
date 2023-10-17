root_dir=dataset
dataset=(Cora)
num_features=(1433)
num_classes=(7)
deltaH=(6)
## DGL
for i in {0..1}
do
    csr_data=${root_dir}/${dataset[i]}/origin/data_csr.pt
    python dgl_sgc.py \
    --data=$csr_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu

    python dgl_gcn.py \
    --data=$csr_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu
    break
done