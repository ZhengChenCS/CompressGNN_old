root_dir=/home/KongmingDataset

dataset=(Cora CiteSeer PubMed ogbn_arxiv)
num_features=(1433 3703 500 128)
num_class=(7 6 3 40)

for i in {0..4}
do 
    # if [ $i -ne 3 ]; then
    # continue
    # fi
    dataset=${dataset[i]}
    graph=$root_dir/$dataset/filter/data.pt
    num_feature=${num_features[i]}
    num_class=${num_class[i]}
    model_path=$root_dir/$dataset/filter/gcn100.pt
    python train.py \
    --model=gcn \
    --is_saved=True \
    --is_test=True \
    --model_path=$model_path \
    --data=$graph \
    --num_features=$num_feature \
    --num_classes=$num_class \
    --epoch=100 \
    --device=gpu
    break
done