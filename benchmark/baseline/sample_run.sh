root_dir=/home/KongmingDataset
# root_dir=/home/KongmingDataset/OGB/dataset

# dataset="cnr-2000  in-2004 eu-2005  hollywood-2009 uk-2007-05@1000000  web-BerkStan"
# dataset="Reddit"

dataset=(Reddit AmazonProducts)
num_features=(602 200)
num_class=(41 107)
gpu_num_part=(32 64)
cpu_num_part=(16 16)

# dataset=(twitter-2010 sk-2005)
# num_features=(52 52)
# num_class=(16 16)
# gpu_num_part=(64 64)
# cpu_num_part=(16 16)

echo "GPU training"
##GPU training epoch:100
for((i=0;i<${#dataset[*]};i++));
do
    if [ $i -ne 1 ]; then
        continue
    fi
    dataset=${dataset[i]}
    num_feature=${num_features[i]}
    num_class=${num_class[i]}
    num_part=${gpu_num_part[i]}
    graph=$root_dir/$dataset/origin/origin.pt
    save_dir=$root_dir/$dataset/origin
    python sample_train.py --model=gcn \
    --data=$graph \
    --num_features=$num_feature \
    --num_classes=$num_class \
    --epoch=100 \
    --device=gpu \
    --num_parts=$num_part \
    --batch_size=1 \
    --save_dir=$save_dir
    break
done

