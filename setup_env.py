#!/usr/bin/env python3
"""
Setup script for fraud detection project environment
Run this script to create virtual environment and install dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def main():
    """Main setup function"""
    print("🚀 Setting up Fraud Detection Project Environment")
    print("=" * 50)
    
    # Check if Python is available
    try:
        python_version = subprocess.run([sys.executable, "--version"], 
                                      capture_output=True, text=True, check=True)
        print(f"📍 Using Python: {python_version.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("❌ Python not found. Please install Python 3.8+ first.")
        sys.exit(1)
    
    # Create virtual environment
    if not os.path.exists(".venv"):
        run_command(f"{sys.executable} -m venv .venv", "Creating virtual environment")
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = ".venv\\Scripts\\activate"
        pip_path = ".venv\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        activate_script = ".venv/bin/activate"
        pip_path = ".venv/bin/pip"
    
    # Upgrade pip in virtual environment
    run_command(f"{pip_path} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        run_command(f"{pip_path} install -r requirements.txt", "Installing project dependencies")
    else:
        print("⚠️  requirements.txt not found. Skipping dependency installation.")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "logs",
        "src",
        "tests",
        "notebooks",
        "reports"
    ]
    
    print("\n🔄 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  📁 {directory}")
    
    # Copy environment template
    if os.path.exists(".env.example") and not os.path.exists(".env"):
        run_command("cp .env.example .env", "Creating environment configuration")
        print("⚠️  Please update .env file with your actual configuration values")
    
    print("\n" + "=" * 50)
    print("🎉 Environment setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    print("2. Update .env file with your configuration")
    print("3. Place your Fraud.csv file in the data/raw/ directory")
    print("4. Start implementing tasks from .kiro/specs/fraud-detection-system/tasks.md")
    print("\n🚀 Happy coding!")

if __name__ == "__main__":
    main()