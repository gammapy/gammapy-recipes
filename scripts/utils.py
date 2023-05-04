# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""PUtility functions that were initially suported with gammapy jupyter."""
import nbformat
import logging
import subprocess
import sys
import time

log = logging.getLogger(__name__)

class BlackNotebook:
    """Manage the process of black formatting.
    Probably this will become available directly in the future.
    See https://github.com/ambv/black/issues/298#issuecomment-476960082
    """

    MAGIC_TAG = "###-MAGIC TAG-"

    def __init__(self, rawnb):
        self.rawnb = rawnb

    def blackformat(self):
        """Format code cells."""
        import black

        for cell in self.rawnb.cells:
            fmt = cell["source"]
            if cell["cell_type"] == "code":
                try:
                    fmt = "\n".join(self.tag_magics(fmt))
                    has_semicolon = fmt.endswith(";")
                    fmt = black.format_str(
                        src_contents=fmt, mode=black.FileMode(line_length=79)
                    ).rstrip()
                    if has_semicolon:
                        fmt += ";"
                except Exception as ex:
                    log.info(ex)
                fmt = fmt.replace(self.MAGIC_TAG, "")
            cell["source"] = fmt

    def tag_magics(self, cellcode):
        """Comment magic commands."""
        lines = cellcode.splitlines(False)
        for line in lines:
            if line.startswith("%") or line.startswith("!"):
                magic_line = self.MAGIC_TAG + line
                yield magic_line
            else:
                yield line


def execute_notebook(path, kernel="python3", loglevel=30):
    """Execute a Jupyter notebook."""
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--allow-errors",
        f"--log-level={loglevel}",
        "--ExecutePreprocessor.timeout=None",
        f"--ExecutePreprocessor.kernel_name={kernel}",
        "--to",
        "notebook",
        "--inplace",
        "--execute",
        f"{path}",
    ]
    t = time.time()
    completed_process = subprocess.run(cmd)
    t = time.time() - t
    if completed_process.returncode:
        log.error(f"Error executing notebook: {path.name} in {path.parent}")
        return False
    else:
        log.info(f"   ... DURATION {path.name}: {t:.1f} seconds")
        return True


def notebook_run(path, kernel="python3"):
    """Execute and parse a Jupyter notebook exposing broken cells."""


    log.info(f"   ... EXECUTING: {path.name} in {path.parent}")
    passed = execute_notebook(path, kernel)
    rawnb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)
    report = ""

    for cell in rawnb.cells:
        if "outputs" in cell.keys():
            for output in cell["outputs"]:
                if output["output_type"] == "error":
                    passed = False
                    traceitems = ["--TRACEBACK: "]
                    for o in output["traceback"]:
                        traceitems.append(f"{o}")
                    traceback = "\n".join(traceitems)
                    infos = "\n\n{} in cell [{}]\n\n" "--SOURCE CODE: \n{}\n\n".format(
                        output["ename"], cell["execution_count"], cell["source"]
                    )
                    report = infos + traceback
                    break
        if not passed:
            break

    if passed:
        log.info(f"   ... PASSED {path.name}")
        return True
    else:
        log.info(f"   ... FAILED {path.name}")
        log.info(report)
        return False


def notebook_black(path):
    """Format code cells with black."""
    rawnb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)
    blacknb = BlackNotebook(rawnb)
    blacknb.blackformat()
    rawnb = blacknb.rawnb
    nbformat.write(rawnb, str(path))
    log.info(f"Applied black to notebook: {path}")

def notebook_strip(path):
    """Strip output cells."""
    rawnb = nbformat.read(str(path), as_version=nbformat.NO_CONVERT)

    rawnb["metadata"].pop("pycharm", None)

    for cell in rawnb.cells:
        if cell["cell_type"] == "code":
            cell["metadata"].pop("pycharm", None)
            cell["metadata"].pop("execution", None)
            cell["execution_count"] = None
            cell["outputs"] = []

    nbformat.write(rawnb, str(path))
    log.info(f"Strip output cells in notebook: {path}")
