"""Export draft_full.md as an IEEE-styled Word document.

The manuscript source of truth is now draft_full.md. This exporter performs a
small Markdown-to-docx conversion so Word output cannot drift from the edited
paper text and tables.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DRAFT = ROOT / "draft_full.md"

_FMT_CANDIDATES = [
    Path(r"c:\Users\12293\Desktop\cpscources\cps3962\project\uav-semantic-mec\scripts\export_ieee_word.py"),
    Path("/mnt/c/Users/12293/Desktop/cpscources/cps3962/project/uav-semantic-mec/scripts/export_ieee_word.py"),
]
_FMT_PATH = next((p for p in _FMT_CANDIDATES if p.is_file()), _FMT_CANDIDATES[0])
_spec = importlib.util.spec_from_file_location("ieee_fmt", _FMT_PATH)
ieee = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ieee)  # type: ignore[union-attr]

Document = ieee.Document
WD_ALIGN_PARAGRAPH = ieee.WD_ALIGN_PARAGRAPH
set_doc_style = ieee.set_doc_style
add_para = ieee.add_para
add_heading = ieee.add_heading
add_figure = ieee.add_figure
add_table = ieee.add_table
add_equation = ieee.add_equation
BODY_FONT_PT = ieee.BODY_FONT_PT
Pt = ieee.Pt


def add_code_block(doc: Document, code_lines: list[str]) -> None:
    if not code_lines:
        return
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run("\n".join(code_lines))
    run.font.name = "Consolas"
    run.font.size = Pt(9)


def _clean_inline(text: str) -> str:
    for ws in ("\\,", "\\;", "\\:", "\\ ", "\\quad", "\\qquad"):
        text = text.replace(ws, " ")
    text = text.replace("\\!", "")
    # Order matters: multi-char commands that share a prefix must precede shorter
    # ones (e.g. \infty and \int before \in) so text.replace does not corrupt them.
    replacements = {
        # font/structure wrappers -> stripped
        r"\mathbb": "", r"\mathcal": "", r"\mathrm": "", r"\mathbf": "",
        r"\text": "", r"\left": "", r"\right": "",
        r"\hat": "", r"\bar": "", r"\tilde": "", r"\vec": "", r"\dot": "",
        # functions
        r"\liminf": "liminf", r"\limsup": "limsup", r"\arg": "arg",
        r"\max": "max", r"\min": "min", r"\exp": "exp",
        r"\log": "log", r"\ln": "ln",
        # relations / operators (longer-prefix first)
        r"\infty": "∞", r"\int": "∫", r"\in": "∈",
        r"\rightarrow": "→", r"\leftarrow": "←", r"\Rightarrow": "⇒",
        r"\leq": "≤", r"\geq": "≥", r"\approx": "≈", r"\propto": "∝",
        r"\gg": "≫", r"\ll": "≪", r"\sim": "~", r"\cdot": "·",
        r"\pm": "±", r"\times": "×", r"\cup": "∪", r"\cap": "∩",
        r"\subseteq": "⊆", r"\emptyset": "∅",
        r"\lfloor": "⌊", r"\rfloor": "⌋", r"\lceil": "⌈", r"\rceil": "⌉",
        r"\sum": "Σ", r"\prod": "∏", r"\ldots": "…", r"\cdots": "⋯",
        r"\top": "T",
        # uppercase Greek
        r"\Delta": "Δ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Omega": "Ω",
        r"\Lambda": "Λ", r"\Theta": "Θ", r"\Phi": "Φ", r"\Psi": "Ψ",
        r"\Gamma": "Γ",
        # lowercase Greek
        r"\mu": "μ", r"\lambda": "λ", r"\kappa": "κ", r"\alpha": "α",
        r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε",
        r"\varphi": "φ", r"\phi": "φ", r"\psi": "ψ", r"\pi": "π",
        r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ", r"\theta": "θ",
        r"\zeta": "ζ", r"\eta": "η", r"\omega": "ω", r"\nu": "ν",
        r"\xi": "ξ",
        # braces written as literals
        r"\{": "{", r"\}": "}", r"\|": "‖",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    for _ in range(3):
        text = re.sub(r"\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _parse_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    rows = []
    for line in lines:
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        rows.append([_clean_inline(p) for p in parts])
    header = rows[0]
    body = [r for r in rows[2:] if len(r) == len(header)]
    return header, body


def _flush_paragraph(doc: Document, buf: list[str]) -> None:
    if not buf:
        return
    text = _clean_inline(" ".join(x.strip() for x in buf).strip())
    if text:
        add_para(doc, text, size=BODY_FONT_PT)
    buf.clear()


def _figure_path(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if p.suffix:
        return p
    for ext in (".png", ".pdf"):
        candidate = p.with_suffix(ext)
        if candidate.exists():
            return candidate
    return p


def _extract_equation_number(latex: str, fallback: int) -> tuple[str, int]:
    tag = re.search(r"\\tag\{([^}]+)\}", latex)
    if not tag:
        return latex.strip(), fallback
    number_text = tag.group(1).strip()
    latex = re.sub(r"\\tag\{[^}]+\}", "", latex).strip()
    try:
        return latex, int(number_text)
    except ValueError:
        return latex, fallback


def _collect_display_equation(lines: list[str], start: int) -> tuple[str, int]:
    """Collect a $$...$$ display equation starting at line start."""
    first = lines[start].strip()
    content = first[2:]
    if content.endswith("$$") and len(content) >= 2:
        return content[:-2].strip(), start + 1

    parts = [content] if content else []
    i = start + 1
    while i < len(lines):
        cur = lines[i].strip()
        if cur.endswith("$$"):
            parts.append(cur[:-2])
            return "\n".join(parts).strip(), i + 1
        parts.append(lines[i])
        i += 1
    return "\n".join(parts).strip(), i


_SKIP_SECTION_PREFIXES = ("INTERNAL", "FIGURE INDEX")
_SKIP_META_PREFIXES = (
    "**Full Manuscript Draft",
    "**Code:**",
    "**Figures:**",
    "**Authors:**",
    "**Affiliation:**",
    "**Email:**",
)

AUTHOR_NAME = "Jiashang Shen"
AUTHOR_AFFIL = "Wenzhou Kean University, Wenzhou, China"
AUTHOR_EMAIL = "shenjias@kean.edu"


def add_author_block(doc: Document) -> None:
    add_para(doc, AUTHOR_NAME, align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
    add_para(doc, AUTHOR_AFFIL, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)
    add_para(doc, AUTHOR_EMAIL, align=WD_ALIGN_PARAGRAPH.CENTER)


def build_paper(doc: Document) -> None:
    lines = DRAFT.read_text(encoding="utf-8").splitlines()
    para_buf: list[str] = []
    eqn = 0
    i = 0
    skip_section = False

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # Drop internal-only sections (claim map, figure index) and their body.
        if stripped.startswith("## "):
            title = _clean_inline(stripped[3:]).upper()
            skip_section = title.startswith(_SKIP_SECTION_PREFIXES)
            if skip_section:
                _flush_paragraph(doc, para_buf)
                i += 1
                continue
        if skip_section:
            i += 1
            continue

        # Drop internal metadata lines from the title block.
        if stripped.startswith(_SKIP_META_PREFIXES):
            i += 1
            continue

        if not stripped or stripped == "---":
            _flush_paragraph(doc, para_buf)
            i += 1
            continue

        if stripped.startswith("```"):
            _flush_paragraph(doc, para_buf)
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            add_code_block(doc, code_lines)
            continue

        if stripped.startswith("$$"):
            _flush_paragraph(doc, para_buf)
            latex, i = _collect_display_equation(lines, i)
            eqn += 1
            latex, number = _extract_equation_number(latex, eqn)
            try:
                add_equation(doc, latex, number)
            except Exception:
                add_para(doc, f"{latex} ({number})", size=BODY_FONT_PT)
            continue

        if stripped.startswith("# "):
            _flush_paragraph(doc, para_buf)
            add_para(
                doc,
                _clean_inline(stripped[2:]),
                bold=True,
                align=WD_ALIGN_PARAGRAPH.CENTER,
                size=16,
            )
            add_author_block(doc)
            i += 1
            continue

        if stripped.startswith("## "):
            _flush_paragraph(doc, para_buf)
            add_heading(doc, _clean_inline(stripped[3:]))
            i += 1
            continue

        if stripped.startswith("### "):
            _flush_paragraph(doc, para_buf)
            add_para(doc, _clean_inline(stripped[4:]), bold=True)
            i += 1
            continue

        if stripped.startswith("!["):
            _flush_paragraph(doc, para_buf)
            m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if m:
                caption, rel = m.groups()
                path = _figure_path(rel)
                if path.exists():
                    add_figure(doc, path, _clean_inline(caption), width=5.8)
                else:
                    add_para(doc, f"[Missing figure: {rel}] {caption}")
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            _flush_paragraph(doc, para_buf)
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            header, body = _parse_table(table_lines)
            add_table(doc, header, body, "")
            continue

        if stripped.startswith("- ") or re.match(r"\d+\.\s+", stripped):
            _flush_paragraph(doc, para_buf)
            add_para(doc, "• " + _clean_inline(re.sub(r"^\d+\.\s+", "", stripped[2:] if stripped.startswith("- ") else stripped)))
            i += 1
            continue

        para_buf.append(stripped)
        i += 1

    _flush_paragraph(doc, para_buf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "T2Sem-DRL_IEEE_Final_JiashangShen.docx"),
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    set_doc_style(doc)
    build_paper(doc)
    doc.save(str(out))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
