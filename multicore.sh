#!/bin/bash

# Caution: Running multiple cores simultaneously may lead to resource contention, potentially affecting the accuracy of results.

source ~/mambaforge/etc/profile.d/conda.sh # Change this path to your conda installation
conda activate base # Change this to the name of your conda environment

FOLDER_PATH="Python Scripts"
ANALYSIS_FOLDER="Analysis"
NUM_RUNS=10
CSV_FILE="$ANALYSIS_FOLDER/run_time.csv"

rm -f "$CSV_FILE"

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

run_multicore() {
    python_files=("$@")

    for file in "${python_files[@]}"; do
        run_python_script "$file" &
    done

    wait
}

cd "$FOLDER_PATH" || exit
python_files=()
for file in *.py; do
    python_files+=("$file")
done

run_multicore "${python_files[@]}"

echo "Execution complete. Results written to $CSV_FILE"

conda deactivate