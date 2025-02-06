#!/bin/bash

source ~/mambaforge/etc/profile.d/conda.sh # Change this path to your conda installation
conda activate base # Change this to the name of your conda environment

PROJECT_FOLDER="$PWD"
FOLDER_PATH="$PROJECT_FOLDER/Python Scripts"
ANALYSIS_FOLDER="$PROJECT_FOLDER/Analysis"
NUM_RUNS=10
CSV_FILE="$ANALYSIS_FOLDER/run_time.csv"

mkdir -p "$ANALYSIS_FOLDER"

if [ -f "$CSV_FILE" ]; then
    rm "$CSV_FILE"
fi

run_python_script() {
    file="$1"
    for ((i=1; i<=$NUM_RUNS; i++)); do
        echo "Running $file - Attempt $i"
        start_time=$(date +%s.%N)
        python "$file"
        end_time=$(date +%s.%N)
        run_time=$(echo "$end_time - $start_time" | bc)
        echo "$file,$run_time" >> "$CSV_FILE"
    done
}

cd "$FOLDER_PATH" || exit
python_files=()
for file in *.py; do
    run_python_script "$file"
done

echo "Execution complete. Results written to $CSV_FILE"

conda deactivate
