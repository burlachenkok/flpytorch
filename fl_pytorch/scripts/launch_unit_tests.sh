#!/usr/bin/env bash

files="utils/buffer.py "\
"utils/worker_thread.py "\
"utils/gpu_utils.py "\
"data_preprocess/read_file_cache.py "\
"utils/thread_pool.py "\
"utils/compressors.py "\
"models/mutils.py "

cd "./../"
#python="python3.9"
python="python"

echo "-----------------------------------------"
echo "Current directory: ${pwd}"
echo "Python path: `which ${python}`"
echo "-----------------------------------------"

for f in ${files}
do
  echo "Process tests from ${f}"
  ${python} -m pytest --disable-pytest-warnings -q ${f}
done
