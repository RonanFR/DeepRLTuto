#!/bin/bash

## Example for running multiplie jobs with different parameters (the jobs are therefore all different) on multiple nodes (in comparison run.sh can only be used to run the exact same
## jobs multiple times on different nodes, not different jobs ob different nodes)

base_name="test_job" # name of what you are running (ex: dqn, etc)
path=/private/home/rfruit/jobs/

for seed in {1..5};do # say the jobs have different seeds
	for lr in 1 2 3; do # and different learning rates
		job_name=${base_name}_seed${seed}_lr${lr}
		mkdir -p ${path}${job_name}/
		slurm_file=${path}${job_name}/${job_name}_run.sh
		echo "#!/bin/bash" >> ${slurm_file}
		echo "#SBATCH --job-name=$job_name" >> ${slurm_file} 
		echo "#SBATCH --output=${path}${job_name}/${job_name}.out" >> ${slurm_file}
		echo "#SBATCH --error=${path}${job_name}/${job_name}.err" >> ${slurm_file}
		echo "#SBATCH --partition=scavenge" >> ${slurm_file}
		echo "#SBATCH --nodes=1" >> ${slurm_file}
		echo "#SBATCH --ntasks-per-node=1" >> ${slurm_file}
		echo "#SBATCH --gres=gpu:1" >> ${slurm_file}
		echo "#SBATCH --cpus-per-task=2" >> ${slurm_file}
		echo "#SBATCH --time=0:0:30" >> ${slurm_file}
		echo "#SBATCH --mail-user=rfruit@fb.com"  >> ${slurm_file}
		echo "#SBATCH --mail-type=fail" >> ${slurm_file}
		echo "#SBATCH --mail-type=end" >> ${slurm_file}
		echo "module purge" >> ${slurm_file}
		echo "module load anaconda3/4.3.1" >> ${slurm_file}
		echo "source activate deepenv2" >> ${slurm_file}
		echo "for name in test1_seed${seed}_lr${lr} test2_seed${seed}_lr${lr} test3_seed${seed}_lr${lr}; do srun --label python ${path}test.py \$name; done" >> ${slurm_file} # what you want to run for ONE JOB, each job consists of a for loop    
		cd ${path}${job_name}/
		sbatch ${slurm_file}
	done
done


