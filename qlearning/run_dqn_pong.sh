#!/bin/bash

base_name="dqn_pong" # name of what you are running (ex: dqn, etc)
path=/private/home/rfruit/DeepRLTuto/qlearning/

for epsd in 10000 100000 200000; do
	for lr in 0.001 0.0001 0.00005; do
		job_name=${base_name}_epsd${epsd}_lr${lr}
		mkdir -p ${path}${job_name}/
		slurm_file=${path}${job_name}/run.sh
		echo "#!/bin/bash" >> ${slurm_file}
		echo "#SBATCH --job-name=$job_name" >> ${slurm_file} 
		echo "#SBATCH --output=${path}${job_name}/${job_name}.out" >> ${slurm_file}
		echo "#SBATCH --error=${path}${job_name}/${job_name}.err" >> ${slurm_file}
		echo "#SBATCH --partition=learnfair" >> ${slurm_file}
		echo "#SBATCH --nodes=2" >> ${slurm_file}
		echo "#SBATCH --ntasks-per-node=2" >> ${slurm_file}
		echo "#SBATCH --gres=gpu:4" >> ${slurm_file}
		echo "#SBATCH --cpus-per-task=5" >> ${slurm_file}
		echo "#SBATCH --time=02:00:00" >> ${slurm_file}
		echo "#SBATCH --mail-user=rfruit@fb.com"  >> ${slurm_file}
		echo "#SBATCH --mail-type=fail" >> ${slurm_file}
		echo "#SBATCH --mail-type=end" >> ${slurm_file}
		echo "module purge" >> ${slurm_file}
		echo "module load anaconda3/4.3.1" >> ${slurm_file}
		echo "source activate deepenv2" >> ${slurm_file}
		task_file=${path}${job_name}/task.sh # New file needed to be able to use SLURM_PROCID
		echo "#!/bin/bash" >> ${task_file}
		echo "task_id=\${SLURM_PROCID}" >> ${task_file}
		chmod u+x ${task_file} # Make the file executable
		info_file=${path}${job_name}/info.txt
		echo "Learning rate: ${lr}" >> ${info_file}
		echo "Epsilon decay last frame: ${epsd}" >> ${info_file}
		echo "for env in \"PongNoFrameskip-v4\"; do python ${path}dqn.py --env \${env} --epsd ${epsd} --lr ${lr} --other \"\${task_id}\"; done" >> ${task_file}
		echo "srun ${task_file}" >> ${slurm_file}
		cd ${path}${job_name}/
		sbatch ${slurm_file}
	done
done


