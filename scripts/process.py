# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Process recipes notebooks for publication."""
import glob
import logging
import nbformat
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from distutils.dir_util import copy_tree
import gammapy
from nbformat.v4 import new_markdown_cell

sys.path.append('scripts')
from utils import notebook_run, notebook_black, notebook_strip


log = logging.getLogger(__name__)
PATH_FILLED = Path(__file__).resolve().parent / ".." / "docs" / "notebooks"
PATH_CLEAN = Path(__file__).resolve().parent / ".." / "docs" / "_static" / "notebooks"
PATH_TEMP = Path(__file__).resolve().parent / ".." / "temp"


def add_box(nb_folder, nb_path):
    """Add box with downloadable links."""

    log.info(f"Adding box in {nb_path}")
    DOWNLOAD_CELL = """
<div class="alert alert-info">

**This is a fixed-text formatted version of a Jupyter notebook**

- **Source files:**
[{nb_filename}](../../_static/notebooks/{nb_folder}/{nb_filename}) |
[{py_filename}](../../_static/notebooks/{nb_folder}/{py_filename})

- **Environment:**
[env.yml](../../_static/notebooks/{nb_folder}/env.yml)
</div>
"""

    # add box
    nb_filename = nb_path.absolute().name
    py_filename = nb_filename.replace("ipynb", "py")
    ctx = dict(
        nb_folder=nb_folder,
        nb_filename=nb_filename,
        py_filename=py_filename,
    )
    strcell = DOWNLOAD_CELL.format(**ctx)
    rawnb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)

    if "nbsphinx" not in rawnb.metadata:
        rawnb.metadata["nbsphinx"] = {"orphan": bool("true")}
        rawnb.cells.insert(0, new_markdown_cell(strcell))
        nbformat.write(rawnb, nb_path)


def clean_notebook(nb_path):
    """Code formatting and strip output."""

    notebook_black(nb_path)

    notebook_strip(nb_path)


def process_notebook(nb_path):
    """Process notebook before html conversion by nbsphinx."""

    if not Path(nb_path).exists():
        log.error(f"File {nb_path} does not exist.")
        return

    # declare paths
    folder_path = Path(os.path.dirname(nb_path))
    folder_name = os.path.basename(folder_path)
    nb_name = nb_path.absolute().name
    nb_temp = PATH_TEMP / folder_name / nb_name
    env_file = folder_path / "env.yml"

    dest_path_temp = PATH_TEMP / folder_name
    dest_path_clean = PATH_CLEAN / folder_name
    dest_path_filled = PATH_FILLED / folder_name

    # clean in recipes
    clean_notebook(nb_path)

    # copy entire recipes content into temp folder
    copy_tree(str(folder_path), str(dest_path_temp))

    # fill it in temp folder
    if notebook_run(nb_temp):

        # copy clean notebook and env.yml in recipes to clean folder
        log.info(f"Copying notebook {nb_path} to {dest_path_clean}")
        os.makedirs(dest_path_clean, exist_ok=True)
        shutil.copy(str(nb_temp), str(dest_path_clean))
        shutil.copy(str(env_file), str(dest_path_clean))

        # conversion to python scripts in clean folder
        static_nb_path = dest_path_clean / nb_name
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "script",
                static_nb_path,
            ]
        )

        # add box in temp
        add_box(folder_name, nb_temp)

        # copy filled notebook and png files in temp to filled folder
        log.info(f"Copying notebook {nb_temp} to {dest_path_filled}")
        os.makedirs(dest_path_filled, exist_ok=True)
        shutil.copy(str(nb_temp), str(dest_path_filled))
        images = list(dest_path_temp.glob("*.png"))
        for im in images:
            shutil.copy(str(im), str(dest_path_filled))
        return True

    else:
        log.error(f"Execution failed for {str(folder_name)}/{nb_name}")
        return False


def tear_down(exitcode):
    if PATH_TEMP.exists():
        shutil.rmtree(PATH_TEMP)
    log.info("Exiting now.")
    sys.exit(exitcode)


def main():
    logging.basicConfig(level=logging.INFO)

    # set up
    PATH_TEMP.mkdir()

    if "GAMMAPY_PATH" not in os.environ:
        log.error("GAMMAPY_PATH environment variable not set.")
        log.error("Running notebook tests requires this environment variable.")
        tear_down(1)

    version = gammapy.__version__
    gammapy_path = os.environ["GAMMAPY_PATH"]
    os.environ.update({"GAMMAPY_DATA": gammapy_path + '/' + version})

    try:
        p = 0
        for idx, arg in enumerate(sys.argv):
            if arg == "--src" and sys.argv[idx+1]:
                src_path = Path(sys.argv[idx+1])
                if not src_path.is_dir():
                    log.error(f"--src argument is not a dir.")
                    tear_down(1)
                p = list(src_path.glob("*.ipynb"))
                if len(p) == 0:
                    log.error(f"Notebook not found in {src_path}")
                    tear_down(1)
                if len(p) > 1:
                    log.error(f"More than one notebook found in {src_path}")
                    tear_down(1)
                if not process_notebook(p[0]):
                    tear_down(1)
        if not p:
            log.error(f"--src argument is needed.")
            tear_down(1)
    except Exception:
        traceback.print_exception(*sys.exc_info())
        tear_down(1)
    else:
        tear_down(0)


if __name__ == "__main__":
    main()
