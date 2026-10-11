"""Save tables and the console log so results can be found again later.

Every table goes to output/tables/ twice:
  * name.csv  to open in Excel or pandas
  * name.tex  to pull into the LaTeX report with \\input{...}
Everything printed to the terminal is also copied into output/run_log.txt.
"""
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "output"
TABLES = OUT / "tables"


def save_table(table, name, index=True):
    """Write one table as csv and tex. `name` is the file name without extension,
    for example "table_8_80items"."""
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / f"{name}.csv", index=index)
    table.to_latex(TABLES / f"{name}.tex", index=index, float_format="%.5f", escape=True)


class Tee:
    """Send printed text to the terminal and to a log file at the same time."""

    def __init__(self, path):
        OUT.mkdir(exist_ok=True)
        self.file = open(path, "w", buffering=1)  # write each line straight away
        self.terminal = sys.stdout

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def start_log(filename="run_log.txt"):
    sys.stdout = Tee(OUT / filename)
