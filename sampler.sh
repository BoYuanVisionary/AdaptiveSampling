#!/bin/bash

# Define the range of values for each parameter
seeds=("1" "2" "3")
dimensions=("1" "10" "20" "100")
# Large condition is boolean, so no need to define values

# Loop through each combination of parameters
for seed in "${seeds[@]}"; do
    for dim in "${dimensions[@]}"; do
        # Determine folder name based on the parameters
        folder_name="output_seed_${seed}_dim_${dim}"
        folder_name="${folder_name}_large"

        # Delete the existing folder (if it exists)
        if [ -d "$folder_name" ]; then
            rm -r "$folder_name"
        fi
        # Create the folder
        mkdir -p "$folder_name"
        # Run the Python script with the current parameters and redirect output
        python samplingMixture.py --seed="$seed" --dimension="$dim" --large_condition --save_images "$folder_name" > "$folder_name/output.txt"
        
        folder_name="output_seed_${seed}_dim_${dim}"
        folder_name="${folder_name}_small"
        if [ -d "$folder_name" ]; then
            rm -r "$folder_name"
        fi
        mkdir -p "$folder_name"
        python samplingMixture.py --seed="$seed" --dimension="$dim" --save_images "$folder_name" >"$folder_name/output.txt"

    done
done
