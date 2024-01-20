hi :)


Password: (Qwerty!123456)

STEP 1 CONNECT TO CLUSTER: SSH into cluster: ssh -X scur1041@snellius.surf.nl

STEP 2 In separate tab not connected to cluster to sync files to cluster: syncdl
rsync -ratulvzP ~/Documents/GitHub/FACT2023 scur1041@snellius.surf.nl:~/


STEP 3 go to project directory (in snellius), submit job, take note of job ID
cd FACT2023/CFC-master
sbatch run_defense_job
# Take note of job ID

STEP 4 Check status of job
squeue -u scur1041
# outputs will be written to ~/slurm-logs/ when job is running
To watch job status:
watch -n 1 squeue -u scur1041
to cancel job: scancel <jobid>
to exit watch: ctrl+c

STEP 5 Copy output files from cluster to local machine
(on local machine:)
rsync -ratulvzP scur1041@snellius.surf.nl:~/FACT2023 ~/Documents/GitHub/

IF IMPATIENT
srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18 --time=02:00:00 --pty bash -i
source activate dl1
cd ../into folder ex. /assignment2/part1
to exit srun: exit 

Disconnect from SSH: exit