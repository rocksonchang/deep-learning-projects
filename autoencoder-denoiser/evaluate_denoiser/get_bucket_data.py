## https://cloud.google.com/storage/docs/gsutil/commands/cp
## Example of using gsutil to copy data from a bucket
## Run this script with: 
## $ python get_buckt_data.py

import os
import argparse

def run_cmd(
            BUCKET="gs://rc_bucket/", 
            BUCKET_FOLDER="mnist_AE_20170824_203440/",
            SRC="",
            DEST="./"
            ):
  ## write and execute shell command
  shell_cmd = "gsutil -m cp -R " + BUCKET + BUCKET_FOLDER + SRC + " " + DEST
  print shell_cmd
  os.system(shell_cmd)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--BUCKET', help="Bucket name.")
  parser.add_argument('--BUCKET_FOLDER', help="Bucket sub-folder name.")
  parser.add_argument('--SRC', help="Source. Leave blank for all files.")
  parser.add_argument('--DEST', help="Destination. Leave blank to specify directory.")

  args = parser.parse_args()
  arguments = args.__dict__
        
  run_cmd(**arguments)