# Setup Instructions

This guide explains how to create a Python virtual environment and install the required dependencies.

```bash
# 1. Create a virtual environment (named .venv)
python3 -m venv .venv

# 2. Activate the environment
# On Linux/MacOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install scikit-learn numpy matplotlib pandas scipy pymc

# 5. Verify installation
python -c "import sklearn, numpy, matplotlib, pandas, scipy, pymc; print('All packages installed successfully!')"


