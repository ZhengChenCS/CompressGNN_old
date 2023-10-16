root_dir=/home/KongmingDataset

dataset=(cnr-2000 web-BerkStan eu-2005 in-2004 uk-2007-05@1000000  hollywood-2009)
num_features=(1024 512 256 256 128 32)
num_class=(8 8 8 8 8 8)

for i in {0..5}
do 
    dataset=${dataset[i]}
    graph=$root_dir/$dataset/compress/data_coo_3.pt
    num_feature=${num_features[i]}
    num_class=${num_class[i]}
    model_path=$root_dir/$dataset/filter/gcn100.pt
    python train.py \
    --model=gcn \
    --data=$graph \
    --is_saved=False \
    --model_path=$model_path \
    --num_features=$num_feature \
    --num_classes=$num_class \
    --epoch=100 \
    --device=gpu
    break
done