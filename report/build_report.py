"""Build report/DATA266_Lab1_Report_Team_PAIR_01.pdf from the Markdown source with pandoc + XeLaTeX.

    python report/build_report.py        (needs pandoc >= 2.x and a TeX Live with xelatex)

Long inline code spans (repository paths) are converted to \\path{...} so that LaTeX can break them across lines.
"""
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "DATA266_Lab1_Report_Team_PAIR_01.md"
OUT = HERE / "DATA266_Lab1_Report_Team_PAIR_01.pdf"

text = SRC.read_text(encoding="utf8")


def convert(m):
    s = m.group(1)
    if len(s) > 22 and ("/" in s or "_" in s) and not any(c in s for c in "%#\\{}"):
        return "\\path{" + s + "}"
    return m.group(0)


text = re.sub(r"`([^`\n]+)`", convert, text)
with tempfile.TemporaryDirectory() as tmp:
    tmp_md = Path(tmp) / SRC.name
    tmp_md.write_text(text, encoding="utf8")
    subprocess.run(["pandoc", str(tmp_md), "-o", str(OUT), "--pdf-engine=xelatex", f"--resource-path={HERE}",
                    "-V", "linestretch=1.05", "-H", str(HERE / "preamble.tex")], check=True)
print("wrote", OUT.name)
