root_dir=/home/KongmingDataset

dataset=(Cora CiteSeer PubMed ogbn_arxiv Reddit AmazonProducts)
num_features=(1433 3703 500 128 602 200)
num_class=(7 6 3 40 41 102)
log_path=./log.txt


for i in {0..5}
do 
    dataset=${dataset[i]}
    graph=$root_dir/$dataset/origin/data_csr.pt
    num_feature=${num_features[i]}
    num_class=${num_class[i]}
    model_path=$root_dir/$dataset/origin/gcn100.pt
    python train.py \
    --data=$graph \
    --num_features=$num_feature \
    --num_classes=$num_class \
    --device=gpu \
    --log_path=$log_path
done