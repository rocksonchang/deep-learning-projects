############################### 
# Launch local training job
# $ ./gcloud.local.run.sh
###############################


# Suppress warning
# "The TensorFlow library wasn't compiled to use FMA instructions, but these 
# are available on your machine and could speed up CPU computations."
# should probably try compiling from source, may get speed-up when using CPU
# see: https://github.com/tensorflow/tensorflow/issues/7778
export TF_CPP_MIN_LOG_LEVEL=2

MODEL="dc_gan"
DATASET="mnist"
export JOB_NAME="${MODEL}_${DATASET}_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR="./tmp/local_train_$JOB_NAME/"

echo ""
echo "job dir=$JOB_DIR"
echo "job name=$JOB_NAME"

mkdir -p $JOB_DIR

gcloud ml-engine local train \
  --module-name trainer.task \
  --package-path ./trainer/ \
  -- \
  --job-dir $JOB_DIR \
  --n_epochs 2 \
  --dataset $DATASET \
  --BATCH_SIZE 128 

#  --job_id $JOB_NAME \


