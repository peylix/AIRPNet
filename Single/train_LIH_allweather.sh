# Stage 1: train the LIH (AIH) module on clean gt images of the allweather dataset.
# LIH only needs clean images — point it at the gt folder of the training set.
#
# Adjust the path below if your data lives elsewhere.
CUDA_VISIBLE_DEVICES=0 python train_LIH.py \
   -d /root/autodl-tmp/allweather/gt \
   -d_test /root/autodl-tmp/allweather/gt \
   --batch-size 16 -lr 1e-4 --save --cuda \
   --exp allweather_lih \
   --num-steps 12 --guide-weight 1 --rec-weight 1 --save-images
