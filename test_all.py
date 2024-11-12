import os
import subprocess

# Define the root directory
root_dir = os.path.abspath(os.getcwd())

# List of subdirectories to process
subdirectories = ["IRT", "IRT_finial", "MIRT", "MIRT_finial", "NCDM", "NCDM_finial"]

# Paths to the Python scripts to execute in each subdirectory
test_ability_script = "test_ability.py"
test_acc_script = "test_acc.py"

# Path to the get_fairness_score.py script
fairness_script = os.path.join(root_dir, "get_fairness_score.py")

def run_script(script_path):
    """Run a Python script and wait for it to complete."""
    try:
        print(f"Running script: {script_path}")
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_path}: {e}")

def main():
    # Run the scripts in each subdirectory
    for subdirectory in subdirectories:
        dir_path = os.path.join(root_dir, subdirectory)

        # Check if the directory exists
        if os.path.isdir(dir_path):
            # Run test_ability.py first
            ability_script_path = os.path.join(dir_path, test_ability_script)
            if os.path.isfile(ability_script_path):
                run_script(ability_script_path)

            # Run test_acc.py second
            acc_script_path = os.path.join(dir_path, test_acc_script)
            if os.path.isfile(acc_script_path):
                run_script(acc_script_path)
        else:
            print(f"Directory {dir_path} does not exist.")

    # Run get_fairness_score.py on the generated v_ability_parameters.csv file
    v_ability_file = os.path.join(root_dir, "v_ability_parameters.csv")
    if os.path.isfile(v_ability_file):
        print(f"Running get_fairness_score.py with input: {v_ability_file}")
        run_script(fairness_script + " " + v_ability_file)
    else:
        print("v_ability_parameters.csv not found. Ensure test_ability.py was executed successfully.")

if __name__ == "__main__":
    main()
