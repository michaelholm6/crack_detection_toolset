import subprocess

#Run this in a miniconda Python 3.11 environment
#For this script, I call it "Python3.11"

def generate_exe():
    command = (
    "conda activate Python3.11 && "
    ".\.venv\Scripts\pyinstaller.exe main.py "
    "--name crack_detection_toolset "
    "--icon molecule.ico "
    "--clean "
    "--add-data model.yml.gz;."
)
    try:
        subprocess.run(command, shell=True, check=True)
        print("Executable generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    generate_exe()