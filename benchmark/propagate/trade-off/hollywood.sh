root_dir=/home/KongmingDataset/cnr-2000/compress/trade_off
max_depth=9
threshold=(4 8 16 32 64 128)

for j in {0..5}
do
    for((k=8; k<=$max_depth; k++))
    do
        compress_coo_data=${root_dir}/coo/${threshold[j]}_${k}.pt
        compress_csr_data=${root_dir}/csr/${threshold[j]}_${k}.pt
        # python analysis.py \
        # --data=$compress_coo_data
        echo "----------Kongming GAS-----------"
        python kongming_gas.py \
        --data=${compress_coo_data}

        # python kongming_gas_t.py \
        # --data=${compress_coo_data} \
        # --device=gpu \
        # --run_nums=100

        # echo "----------Kongming SPMM-----------"
        # python kongming_spmm.py \
        # --data=${compress_csr_data}

        # python kongming_spmm_t.py \
        # --data=${compress_csr_data} \
        # --device=gpu \
        # --run_nums=100
        break
    done
    break
done
