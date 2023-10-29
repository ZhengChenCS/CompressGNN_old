root_dir=/home/KongmingDataset

dataset=(cnr-2000 web-BerkStan eu-2005 in-2004 uk-2007-05@1000000 hollywood-2009)

for i in {0..5}
do
    echo ${dataset[i]}
    python createTorchDataset.py $root_dir/${dataset[i]} coo
    python createTorchDataset.py $root_dir/${dataset[i]} csr
done

vaild_dataset=(Cora CiteSeer PubMed Reddit AmazonProducts)
for i in {0..4}
do 
    echo ${vaild_dataset[i]}
    python createTorchDataset.py $root_dir/${vaild_dataset[i]} coo
    python createTorchDataset.py $root_dir/${vaild_dataset[i]} csr
done