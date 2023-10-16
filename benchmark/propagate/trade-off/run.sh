root_dir=/home/KongmingDataset

dataset=(cnr-2000 web-BerkStan eu-2005 in-2004 uk-2007-05@1000000 hollywood-2009 Reddit AmazonProducts)
depth=(3 4 5 6 7 8)
threshold=(4 8 16 32 64 128)
for i in {0..0}
do
    echo ${dataset[i]}
    for j in {0..0}
    do
        for k in {0..5}
        do
            echo ${threshold[j]}_${depth[k]}.pt
            compress_coo_data=${root_dir}/${dataset[i]}/compress/trade_off/coo/${threshold[j]}_${depth[k]}.pt
            compress_csr_data=${root_dir}/${dataset[i]}/compress/trade_off/csr/${threshold[j]}_${depth[k]}.pt
            echo "----------Kongming GAS-----------"
            # python kongming_gas.py \
            # --data=${compress_coo_data}

            python kongming_gas_t.py \
            --data=${compress_coo_data} \
            --device=gpu \
            --run_nums=100

            # echo "----------Kongming SPMM-----------"
            # python kongming_spmm.py \
            # --data=${compress_csr_data}

            # python kongming_spmm_t.py \
            # --data=${compress_csr_data} \
            # --device=gpu \
            # --run_nums=100
        done
    done
done


# for i in {6..7}
# do
#     echo ${dataset[i]}
#     for j in {0..5}
#     do
#         for k in {0..5}
#         do
#             echo ${threshold[j]}_${depth[k]}.pt
#             compress_coo_data=${root_dir}/${dataset[i]}/compress/trade_off/coo/${threshold[j]}_${depth[k]}.pt
#             compress_csr_data=${root_dir}/${dataset[i]}/compress/trade_off/csr/${threshold[j]}_${depth[k]}.pt
#             echo "----------Kongming SPMM-----------"
#             python kongming_spmm.py \
#             --data=${compress_csr_data}

#             python kongming_spmm_t.py \
#             --data=${compress_csr_data} \
#             --device=gpu \
#             --run_nums=100
#         done
#     done
# done
