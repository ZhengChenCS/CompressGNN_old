root_dir=dataset
dataset=(Cora)
num_features=(1433)
num_classes=(7)
deltaH=(6)

# compressgnn
for i in {0..5}
do
    coo_data=${root_dir}/${dataset[i]}/compress/data_coo_1.pt
    csr_data=${root_dir}/${dataset[i]}/compress/data_csr_1.pt
    python compressgnn_sgc.py \
    --data=$coo_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu \
    --deltaH=${deltaH[i]}

    python compressgnn_sgc.py \
    --data=$csr_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu \
    --deltaH=${deltaH[i]}

    python compressgnn_gcn.py \
    --data=$coo_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu \
    --deltaH=${deltaH[i]}

    python compressgnn_gcn.py \
    --data=$csr_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu \
    --deltaH=${deltaH[i]}
    break
done
