"""Inline the SVGs of `report_figures` into the status page.

The figures are INLINE rather than <img src>: the page has to survive being sent as one
file, and a status page whose plots silently 404 on the reader's machine is worse than
no page. SVG rather than PNG so the axis labels stay readable at any zoom.

Run `report_figures.py` first; this script only assembles.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PLACEHOLDER = re.compile(r"\{\{fig:([a-z0-9_]+)\}\}")


def inline(svg: str) -> str:
    """Strip the XML prolog and DOCTYPE, keep the <svg> element itself.

    Both are illegal inside an HTML body, and browsers that tolerate them do so by
    dropping the rest of the document -- a failure that looks like a blank page.
    """
    i = svg.index("<svg")
    return svg[i:]


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, default=here / "report_page.tmpl.html")
    ap.add_argument("--figures", type=Path, default=here / "figures" / "report")
    ap.add_argument("--out", type=Path, default=here / "figures" / "report_page.html")
    args = ap.parse_args()

    html = args.template.read_text()
    missing: list[str] = []

    def sub(m: re.Match) -> str:
        name = m.group(1)
        p = args.figures / f"{name}.svg"
        if not p.exists():
            missing.append(name)
            return m.group(0)
        return f'<figure id="{name}">\n{inline(p.read_text())}\n</figure>'

    html = PLACEHOLDER.sub(sub, html)
    if missing:
        raise SystemExit(f"no SVG for: {', '.join(missing)} -- run report_figures.py")

    args.out.write_text(html)
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
