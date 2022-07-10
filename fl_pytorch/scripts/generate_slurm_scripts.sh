#!/usr/bin/env bash

export datasets=("cifar10_fl")
export g_learning_rates=("0.1" "0.06" "0.03" "0.01" "0.006" "0.003" "0.001" "0.0006" "0.0003" "0.0001" "0.00006" "0.00003" "0.00001")

export wandb_key=""

if [[ ! -f "$1" ]]
then
   echo "File '$1' with command line does not exist. Press any key to continue."
   read
   exit -1
fi

for dataset in "${datasets[@]}"
do
for g_lr in "${g_learning_rates[@]}"
do
   export job_id=$(($RANDOM))

   export dataset
   export g_lr

   fname_with_script=$1
   dest_fname=ibex_launch_scipt_${dataset}_${g_lr}_${fname_with_script%.*}_${job_id}.sbatch
   cat boilerplate.sbatch > ${dest_fname} 
   echo -n "srun " >> ${dest_fname} 
   envsubst <${fname_with_script} >>${dest_fname}
done
done

echo "Completed successfully"
