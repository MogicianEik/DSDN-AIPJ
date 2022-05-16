for i in 0 1 2 3 4
do
    export CUDA_VISIBLE_DEVICES=0
    python -W igonore src/main.py \
    --n_class 2 \
    --data_path "data/" \
    --eval_file "data/test_$i.txt" \
    --model_path "outputs/saved_models/" \
    --log_path "outputs/runs/" \
    --task_name "testing_fold$i" \
    --mode 2 \
    --batch_size 1 \
    --sub_batch_size 6 \
    --size_g 508 \
    --size_p 224 \
    --path_g "global_only_$i.pth" \
    --path_g2l "global2local_$i.pth" \
    --evaluation \
    --visualization
done
