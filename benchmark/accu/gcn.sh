root_dir=/home/KongmingDataset

dataset=(Cora CiteSeer PubMed ogbn_arxiv Reddit AmazonProducts)
num_features=(1433 3703 500 128 602 200)
num_class=(7 6 3 40 41 102)
TrueFalse=(True False)
log_path=./benchmark.txt

for i in {0..5}
do 
    for index_cache in True False
    do
        for deltaH in {2..20..3}
        do
            for reset_step in 1 4 16 100
            do
                dataset=${dataset[i]}
                graph=$root_dir/$dataset/origin/data_csr.pt
                # graph=$root_dir/$dataset/origin/data_coo.pt
                num_feature=${num_features[i]}
                num_class=${num_class[i]}
                model_path=$root_dir/$dataset/origin/gcn100.pt
                python train.py \
                --model=gcn \
                --is_saved=False \
                --is_test=True \
                --model_path=$model_path \
                --data=$graph \
                --num_features=$num_feature \
                --num_classes=$num_class \
                --epoch=100 \
                --device=gpu \
                --deltaH=$deltaH \
                --index_cache=$index_cache \
                --reset_step=$reset_step \
                --log_path=$log_path
            done
        done
    done
done