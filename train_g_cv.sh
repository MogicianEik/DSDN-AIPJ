export CUDA_VISIBLE_DEVICES=0
for i in 0 1 2 3 4
do
    python src/main.py \
    --n_class 2 \
    --data_path "data/" \
    --model_path "outputs/saved_models/" \
    --train_file "data/train_$i.txt" \
    --eval_file "data/val_$i.txt" \
    --log_path "outputs/runs/" \
    --task_name "global_only_$i" \
    --mode 1 \
    --batch_size 5 \
    --sub_batch_size 6 \
    --size_g 508 \
    --size_p 508
done
