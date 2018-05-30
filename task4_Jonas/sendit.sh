#! /bin/bash

time="24:00" # which queue?
mail="-N" # empty otherwise

# make sure proper module is loaded
module load python/3.6.0

# Read input parameters from params.dat
# If we reached the last line (that just contains FIN) stop the script
while read line; do
  if [ "$line" == "FIN" ]; then
    break
  fi
  # Get checkpt* as jobname
  jobname=$(echo $line | awk '{print $1;}')
  # Submit euler job
  bsub -W $time $mail -J $jobname python3 my_ladder.py $line
done <params.dat
