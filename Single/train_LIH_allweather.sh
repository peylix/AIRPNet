# Stage 1: train the LIH (AIH) module on clean gt images of the allweather dataset.
# LIH only needs clean images — point it at the gt folder of the training set.
#
# Notes:
#  - --save-images is intentionally OFF: it writes ~8 PNGs per test image and
#    will fill the disk (especially with autodl-tmp). Re-enable only when you
#    want to inspect a small held-out test set.
#  - -d_test points to a small held-out gt dir to keep validation fast.
CUDA_VISIBLE_DEVICES=0 python train_LIH.py \
   -d /root/autodl-tmp/allweather/gt \
   -d_test /root/autodl-tmp/allweather/gt_val \
   --batch-size 16 --test-batch-size 2 -lr 1e-4 --save --cuda \
   --exp allweather_lih \
   --num-steps 12 --guide-weight 1 --rec-weight 1 \
   -e 200
