import subprocess
import sys

subprocess.run([sys.executable, "regenerate_data.py"])
subprocess.run([sys.executable, "recompute_fourier_models.py"])
subprocess.run([sys.executable, "recompute_AR_models.py"])
subprocess.run([sys.executable, "obtain_probabilities.py"])
subprocess.run([sys.executable, "run_diagnostics.py"])