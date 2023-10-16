root_dir=/home/KongmingDataset
# root_dir=/home/KongmingDataset/OGB/dataset

dataset=(Reddit AmazonProducts)
num_features=(602 200)
num_class=(41 107)
gpu_num_part=(32 64)
cpu_num_part=(16 16)
batch_size=(-1 400000)

echo "GPU training"
##GPU training epoch:100
for((i=0;i<${#dataset[*]};i++));
do
    # if [ $i -ne 1 ]; then
    #     continue
    # fi
    dataset=${dataset[i]}
    num_feature=${num_features[i]}
    num_class=${num_class[i]}
    num_part=${gpu_num_part[i]}
    bsize=${batch_size[i]}
    graph=$root_dir/$dataset/origin/origin.pt
    save_dir=$root_dir/$dataset/origin
    sgc_path=$root_dir/$dataset/origin/SGCData.pt
    python sample_train.py --model=sgc \
    --data=$graph \
    --num_features=$num_feature \
    --num_classes=$num_class \
    --epoch=100 \
    --device=gpu \
    --num_parts=$num_part \
    --sgc_batch_size=$bsize \
    --batch_size=1 \
    --save_dir=$save_dir\
    --sgc_path=$sgc_path
    break
done
