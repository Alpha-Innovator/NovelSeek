python experiment.py \
  --batch_size 128 \
  --num_workers 4 \
  --out_dir $1 \
  --in_channels 3 \
  --data_root /path/to/data_root \
  --learning_rate 0.1 \
  --max_epoch 200 \
  --val_per_epoch 5