root_dir=/home/KongmingDataset

dataset=(cnr-2000 web-BerkStan eu-2005 in-2004 uk-2007-05@1000000  hollywood-2009 Reddit AmazonProducts)
num_features=(1024 512 256 256 128 32 602 200)
num_class=(8 8 8 8 8 8 41 107)

for i in {0..0}
do 
    dataset=${dataset[i]}
    graph=$root_dir/$dataset/origin/data_csr.pt
    # graph=$root_dir/$dataset/origin/data_coo.pt
    num_feature=${num_features[i]}
    num_class=${num_class[i]}
    model_path=$root_dir/$dataset/origin/gcn100.pt
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