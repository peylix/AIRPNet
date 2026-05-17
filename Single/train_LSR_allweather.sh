# Stage 2: train the LSR (ASR) module on the paired all-weather dataset.
# Training data is mixed (rain + raindrop + snow) under one root with input/ and gt/.
# Testing data lives in three separate roots with different subdir names; each
# test set is evaluated independently and metrics are logged per task.
#
# Adjust the paths below before running.
CUDA_VISIBLE_DEVICES=0 python train_LSR_allweather.py \
   -d /root/autodl-tmp/allweather \
   --rainhaze_test /root/autodl-tmp/CVPR19RainTrain/test \
   --raindrop_test /root/autodl-tmp/raindrop_data/test_a \
   --snow_test     /root/autodl-tmp/Snow100K-testset/jdway/GameSSD/overlapping/test/Snow100K-L \
   --batch-size 16 --val_freq 50 -lr 1e-4 --save --cuda \
   --exp allweather_lsr \
   --nafwidth 32 --mid 2 --enc 2 2 4 --dec 2 2 2 \
   --klvl 3 --steps 4 --num_step 12 \
   --patch-size 224 224 --test-patch-size 256 256 \
   --cweight 1 --sweight 7 --pweight_c 0.005 \
   --save_img \
   --hide_checkpoint experiments/allweather_lih/checkpoints/_checkpoint_best_loss.pth.tar
