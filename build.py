import subprocess
import sys
import os
import venv
import shutil

"""Building with conda can cause errors in the final program"""

def create_virtual_env(env_dir):
    """Create a virtual environment."""
    print("Creating a virtual environment...")
    venv.create(env_dir, with_pip=True)
    print(f"Virtual environment created at {env_dir}")


def install_requirements(env_dir):
    """Install the required libraries from requirements.txt in the virtual environment."""
    print("Installing dependencies in the virtual environment...")
    pip_path = os.path.join(env_dir, "Scripts", "pip") if os.name == "nt" else os.path.join(env_dir, "bin", "pip")
    result = subprocess.run(
        [pip_path, "install", "-r", "requirements.txt"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        print("Dependencies installed successfully in the virtual environment!")
    else:
        print("Failed to install dependencies:")
        print(result.stderr)
        sys.exit(1)


def build_app(env_dir):
    """Build the application using PyInstaller from the virtual environment."""
    print("Building the application...")

    # Get absolute paths for files and folders
    base_dir = os.getcwd()
    templates_path = os.path.join(base_dir, "frontend", "templates")
    static_path = os.path.join(base_dir, "frontend", "static")
    diagnosis_path = os.path.join(base_dir, "psychDiagnosis")
    icon_path = os.path.join(base_dir, "icon.ico")
    app_file = os.path.join(base_dir, "app.py")

    # Ensure paths exist
    for path in [templates_path, static_path, diagnosis_path, icon_path, app_file]:
        if not os.path.exists(path):
            print(f"Error: Required path not found - {path}")
            sys.exit(1)

    # PyInstaller path in virtual environment
    pyinstaller_path = os.path.join(env_dir, "Scripts", "pyinstaller") if os.name == "nt" else os.path.join(env_dir, "bin", "pyinstaller")

    # PyInstaller command
    command = [
        pyinstaller_path,
        "--onefile",
        "--windowed",
        "--hidden-import=openpyxl.cell._writer",
        "--add-data", f"{templates_path};frontend/templates",
        "--add-data", f"{static_path};frontend/static",
        "--add-data", f"{diagnosis_path};psychDiagnosis",
        "--name", "PsychopathyDiagnosisApp",
        "--icon", icon_path,
        app_file
    ]

    # Run the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print("Build successful!")
    else:
        print("Build failed:")
        print(result.stderr)
        sys.exit(1)


def cleanup(env_dir):
    """Remove the virtual environment directory."""
    print(f"Cleaning up virtual environment at {env_dir}...")
    try:
        shutil.rmtree(env_dir)
        print("Virtual environment removed successfully.")
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    try:
        env_dir = os.path.join(os.getcwd(), "build_env")
        create_virtual_env(env_dir)
        install_requirements(env_dir)
        build_app(env_dir)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        cleanup(env_dir)
