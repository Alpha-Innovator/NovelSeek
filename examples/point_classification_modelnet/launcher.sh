python experiment.py \
  --batch_size 64 \
  --out_dir $1 \
  --in_channels 6 \
  --num_points 1024 \
  --num_category 40 \
  --data_root /path/to/data \
  --learning_rate 1e-3 \
  --max_epoch 200 \
  --val_per_epoch 5

 #/path/to/data_root \