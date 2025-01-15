import subprocess
import sys

def install_requirements():
    """Install the required libraries from requirements.txt."""
    print("Installing dependencies from requirements.txt...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print("Dependencies installed successfully!")
    else:
        print("Failed to install dependencies:")
        print(result.stderr)
        sys.exit(1)

def build_app():
    """Build the application using PyInstaller."""
    print("Building the application...")
    command = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--hidden-import=openpyxl.cell._writer",
        "--add-data", "frontend/templates;frontend/templates",
        "--add-data", "frontend/static;frontend/static",
        "--add-data", "psychDiagnosis;psychDiagnosis",
        "--name", "PsychopathyDiagnosisApp",
        "--icon", "icon.ico",
        "app.py"
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        print("Build successful!")
    else:
        print("Build failed:")
        print(result.stderr)
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
    build_app()
