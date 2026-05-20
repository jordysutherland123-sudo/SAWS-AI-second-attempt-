import subprocess
from pathlib import Path
import sys


def create_venv(venv_path: str = ".venv") -> Path:
    """Create a local virtual environment folder."""
    venv_dir = Path(venv_path)
    if not venv_dir.exists():
        print(f"Creating virtual environment at {venv_dir}")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    else:
        print(f"Virtual environment already exists at {venv_dir}")
    return venv_dir


def install_requirements(venv_path: str = ".venv", requirements_file: str = "requirements.txt") -> None:
    """Install Python dependencies into the virtual environment."""
    pip_executable = Path(venv_path) / "Scripts" / "pip.exe"
    if not pip_executable.exists():
        raise FileNotFoundError(f"pip not found in virtual environment at {pip_executable}")
    print(f"Installing requirements from {requirements_file}")
    subprocess.run([str(pip_executable), "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(pip_executable), "install", "-r", requirements_file], check=True)


if __name__ == "__main__":
    env_dir = create_venv()
    install_requirements(str(env_dir))
    print("Environment setup complete.")
