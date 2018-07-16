squeue -u ychandak

sacct -u ychandak  --format=JobID,JobName,MaxRSS,Elapsed,AllocCPUS,NTasks,State

scancel -u ychandak
