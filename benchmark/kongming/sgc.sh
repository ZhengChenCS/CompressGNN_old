root_dir=/home/KongmingDataset

dataset=(cnr-2000 web-BerkStan eu-2005 in-2004 uk-2007-05@1000000  hollywood-2009)
num_features=(1024 512 256 256 128 32)
num_class=(8 8 8 8 8 8)

for((i=0;i<${#dataset[*]};i++));
do 
    dataset=${dataset[i]}
    graph=$root_dir/$dataset/filter/data.pt
    num_feature=${num_features[i]}
    num_class=${num_class[i]}
    python train.py \
    --model=sgc \
    --data=$graph \
    --num_features=$num_feature \
    --num_classes=$num_class \
    --epoch=100 \
    --device=gpu \
    --sgc_path=$root_dir/$dataset/filter/sgc.pt
    break
done