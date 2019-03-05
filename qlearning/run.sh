#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=first_test_slurm
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/private/home/%u/jobs/sample-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/private/home/%u/jobs/sample-%j.err

## partition name: uninterrupted (default), scavenge (free to try any crazy ideas, slower to get the job done), learnfair?
#SBATCH --partition=uninterrupted
## number of nodes
#SBATCH --nodes=1

## number of tasks per node
#SBATCH --ntasks-per-node=1

## number of gpus per node
#SBATCH --gres=gpu:1

## number of CPUs per task
#SBATCH --cpus-per-task=2 

## Time needed hours:minutes:seconds
#SBATCH --time=0:0:30

## Send notification by email
#SBATCH --mail-user=rfruit@fb.com
#SBATCH --mail-type=fail # mail once the job fails
#SBATCH --mail-type=end # mail once the job finishes


### Section 2: Setting environment variables for the job
### Remember that all the module command does is set environment
### variables for the software you need to. Here I am assuming I
### going to run something with python.
### You can also set additional environment variables here and
### SLURM will capture all of them for each task
# Start clean
module purge

# Load what we need
module load anaconda3/4.3.1
source activate deepenv2


### Section 3:
### Run your job. Note that we are not passing any additional
### arguments to srun since we have already specificed the job
### configuration with SBATCH directives
### This is going to run ntasks-per-node x nodes tasks with each
### task seeing all the GPUs on each node. However I am using
### the wrapper.sh example I showed before so that each task only
### sees one GPU
for name in test1 test2 test3
do
 	srun --label python test.py $name
done	
