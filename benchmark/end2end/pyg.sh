root_dir=dataset
dataset=(Cora)
num_features=(1433)
num_classes=(7)
deltaH=(6)

# PYG
for i in {0..5}
do
    coo_data=${root_dir}/${dataset[i]}/origin/data_coo.pt
    csr_data=${root_dir}/${dataset[i]}/origin/data_csr.pt
    python pyg_sgc.py \
    --data=$coo_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu

    python pyg_sgc.py \
    --data=$csr_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu

    python pyg_gcn.py \
    --data=$coo_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu

    python pyg_gcn.py \
    --data=$csr_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu
    break
done

# for i in {6..7}
# do
#     csr_data=${root_dir}/${dataset[i]}/origin/data_csr.pt
#     python pyg_sgc.py \
#     --data=$csr_data \
#     --num_features=${num_features[i]} \
#     --num_classes=${num_classes[i]} \
#     --epochs=100 \
#     --device=gpu
#     python pyg_gcn.py \
#     --data=$csr_data \
#     --num_features=${num_features[i]} \
#     --num_classes=${num_classes[i]} \
#     --epochs=100 \
#     --device=gpu
# done