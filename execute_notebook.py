"""Execute the notebook with Jupyter runtime files kept inside the project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".jupyter-runtime")
os.environ["IPYTHONDIR"] = str(ROOT / ".ipython")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")
for directory in [".jupyter-runtime", ".ipython", ".matplotlib"]:
    (ROOT / directory).mkdir(exist_ok=True)

import nbformat
from nbclient import NotebookClient

path = ROOT / "telco_churn.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(notebook, path)
print("Notebook executed and saved successfully")
