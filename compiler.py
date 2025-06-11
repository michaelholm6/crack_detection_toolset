import subprocess
import PyInstaller.__main__

#Run this in a miniconda Python 3.11 environment
#For this script, I call it "Python3.11"

def generate_exe():

    PyInstaller.__main__.run([
        "main.py",
        "--name", "crack_detection_toolset",
        "--icon", "molecule.ico",
        "--add-data", "model.yml.gz;.",
    ])

if __name__ == "__main__":
    generate_exe()
    