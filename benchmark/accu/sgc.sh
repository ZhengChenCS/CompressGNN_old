root_dir=/home/KongmingDataset

dataset=(Cora CiteSeer PubMed ogbn_arxiv Reddit AmazonProducts)
num_features=(1433 3703 500 128 602 200)
num_class=(7 6 3 40 41 102)

for i in {0..3}
do 
    for deltaH in {2..4}
        do
        dataset=${dataset[i]}
        graph=$root_dir/$dataset/origin/data_csr.pt
        # graph=$root_dir/$dataset/origin/data_coo.pt
        num_feature=${num_features[i]}
        num_class=${num_class[i]}
        model_path=$root_dir/$dataset/origin/gcn100.pt
        python train.py \
        --model=sgc \
        --data=$graph \
        --num_features=$num_feature \
        --num_classes=$num_class \
        --epoch=100 \
        --device=gpu \
        --sgc_path=$root_dir/$dataset/origin/sgc.pt \
        --deltaH=$deltaH
    done
done