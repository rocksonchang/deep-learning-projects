############################### 
# Launch remote training job
# $ ./gcloud.remote.run.sh
###############################


# Suppress warning
# "The TensorFlow library wasn't compiled to use FMA instructions, but these 
# are available on your machine and could speed up CPU computations."
# should probably try compiling from source, may get speed-up when using CPU
# see: https://github.com/tensorflow/tensorflow/issues/7778
export TF_CPP_MIN_LOG_LEVEL=2

MODEL="dcgan"
DATASET="mnist"
export BUCKET_NAME="rc_bucket" 
export JOB_NAME="${MODEL}_${DATASET}_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME/
export REGION=us-east1
export PATH_TO_DEPENDENCIES=gs://$BUCKET_NAME

echo ""
echo "job dir = $JOB_DIR"
echo "job name = $JOB_NAME"

gcloud ml-engine jobs submit training $JOB_NAME \
  --module-name trainer.task \
  --job-dir $JOB_DIR \
  --package-path ./trainer/ \
  --region $REGION \
  --config=trainer/cloudml-gpu.yaml \
  -- \
  --n_epochs 50 \
  --dataset $DATASET \
  --BATCH_SIZE 128 

#--job_id $JOB_NAME \