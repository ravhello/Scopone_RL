import os

# Set the source folder
source_folder = os.path.dirname(__file__)  # Use the script folder
output_file = "all_code.py"

# Get the name of the current script
current_script = os.path.basename(__file__)

# Open the output file for writing
with open(output_file, "w", encoding="utf-8") as outfile:
    # Scan all files in the folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".py") and filename not in {current_script, output_file, "run_profiling.py", "json_visual.py", "comparison_batch.py", "grad_accum_comparison.py", "visualize_database_trials.py", "all_code2.py"}:  # Exclude itself, the output file, and the specified files
            file_path = os.path.join(source_folder, filename)
            with open(file_path, "r", encoding="utf-8") as infile:
                outfile.write(f"# --- Content of {filename} ---\n\n")
                outfile.write(infile.read() + "\n\n")

print(f"Code copied to {output_file}")
