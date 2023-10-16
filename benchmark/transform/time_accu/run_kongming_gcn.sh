root_dir=/home/KongmingDataset

dataset=(Reddit)
num_features=(1433 3703 500 602 200)
num_classes=(7 6 3 41 102) 
deltaH=(6 6 9 0 5)


for i in 1 2 4 8 16 32
do
    data=${root_dir}/Reddit/origin/data_csr.pt
    python kongming_gcn.py \
    --data=${data} \
    --num_features=602 \
    --num_classes=41 \
    --epochs=200 \
    --device=gpu \
    --deltaH=0 \
    --reset=5 \
    --cache \
    --index_cache
    break
done
