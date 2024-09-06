#!/bin/bash

# Define the number of times you want to run the Python script
NUM_RUNS=10

# Path to your Python script
PYTHON_SCRIPT="/home/cha/isaac_ws/AMP_for_hardware/legged_gym/scripts/train.py"
TASK_NAME="tocabi_amp_rand"

# Loop to run the Python script multiple times
for (( i=1; i<=NUM_RUNS; i++ ))
do
  echo "Running iteration $i"
  
  # Check if this is the first iteration
  if [ $i -eq 1 ]; then
    python $PYTHON_SCRIPT --task=$TASK_NAME --headless
  else
    python $PYTHON_SCRIPT --task=$TASK_NAME --headless --resume
  fi

  # Check if the script exited with a non-zero status
  if [ $? -ne 0 ]; then
    echo "Script failed on iteration $i. Exiting..."
    exit 1
  fi
  
  echo "Iteration $i completed successfully"
done

echo "All iterations completed."
