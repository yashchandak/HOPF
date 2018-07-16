#!/usr/bin/env bash

## Login
ssh ychandak@swarm2.cs.umass.edu

## activate pytorch
source activate pytorch

## Sync files
git stash
git clean -nd # see what files will get removed
git clean -fd # remove those files/changes
git pull

## Create the new required directories
mkdir Experiments
mkdir swarm/stdoutput

## Cythonize all the files
python cython_all_setup.py

##TODO: better to do a demo run first, which would also ensure that all common folders are created.

## Submit the job finally
sh submitJobs.sh

## check number of programs running
squeue -u ychandak | wc -l

##Cross check the cluster usage
#This shows a lot of lines like this, showing some of the jobs currently running on 2 cores (the minimum is 2 because of hyperthreading).
squeue -l --Format="account,username,numcpus,state,timeleft,jobid,memory"
#To get information about one using the job ID:
# ssh into the corresponding job node and run htop

## collect results
scp -r ychandak@swarm2.cs.umass.edu:~/RL/Experiments ./exp_swarm