root_dir=/home/KongmingDataset

dataset=(cnr-2000 web-BerkStan eu-2005 in-2004 uk-2007-05@1000000 hollywood-2009 Reddit AmazonProducts)
num_features=(1024 512 256 256 128 32 602 200)
num_classes=(8 16 8 8 8 40 41 107)
deltaH=(5 5 5 5 5 5 0 5)

# kongming
for i in {0..5}
do
    coo_data=${root_dir}/${dataset[i]}/compress/data_coo_3.pt
    csr_data=${root_dir}/${dataset[i]}/compress/data_csr_3.pt
    python kongming_sgc.py \
    --data=$coo_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu \
    --deltaH=${deltaH[i]}

    python kongming_sgc.py \
    --data=$csr_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu \
    --deltaH=${deltaH[i]}

    python kongming_gcn.py \
    --data=$coo_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu \
    --deltaH=${deltaH[i]}

    python kongming_gcn.py \
    --data=$csr_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu \
    --deltaH=${deltaH[i]}
done

for i in {6..7}
do
    csr_data=${root_dir}/${dataset[i]}/compress/data_csr_3.pt
    python kongming_sgc.py \
    --data=$csr_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu \
    --deltaH=${deltaH[i]}
    python kongming_gcn.py \
    --data=$csr_data \
    --num_features=${num_features[i]} \
    --num_classes=${num_classes[i]} \
    --epochs=100 \
    --device=gpu \
    --deltaH=${deltaH[i]}
done