#!/usr/bin/env python3
"""Build the preliminary technical report PDF from technical_report.md."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab import rl_config
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "technical_report.md"
OUTPUT = ROOT / "technical_report.pdf"
PAGE_WIDTH, PAGE_HEIGHT = A4

# Stable timestamps/object identifiers make repeated builds byte-for-byte
# identical when the Markdown source and dependencies are unchanged.
rl_config.invariant = 1


def inline_markup(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(
        r"\[(.+?)\]\((https?://[^)]+)\)",
        r"<link href='\2' color='#2457A6'><u>\1</u></link>",
        text,
    )
    return text


def architecture_diagram() -> Drawing:
    d = Drawing(470, 190)
    navy = colors.HexColor("#17365D")
    blue = colors.HexColor("#DCE6F1")
    green = colors.HexColor("#E2F0D9")
    amber = colors.HexColor("#FFF2CC")

    def box(x, y, w, h, title, subtitle, fill):
        d.add(Rect(x, y, w, h, 7, 7, fillColor=fill, strokeColor=navy, strokeWidth=1))
        d.add(String(x + w / 2, y + h - 17, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=9, fillColor=navy))
        d.add(String(x + w / 2, y + 12, subtitle, textAnchor="middle", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#444444")))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=navy, strokeWidth=1.2))
        d.add(Polygon([x2, y2, x2 - 6, y2 + 3, x2 - 6, y2 - 3], fillColor=navy, strokeColor=navy))

    box(5, 120, 98, 46, "Sensor input", "acc + gyro", blue)
    box(125, 120, 98, 46, "Preprocessing", "25 Hz + validity", green)
    box(245, 120, 98, 46, "Recognition", "7 classes/window", amber)
    box(365, 120, 98, 46, "Timeline", "intervals + features", green)
    arrow(103, 143, 125, 143)
    arrow(223, 143, 245, 143)
    arrow(343, 143, 365, 143)
    box(45, 35, 105, 46, "Natural query", "one question", blue)
    box(190, 35, 110, 46, "Task router", "fast path / slow path", amber)
    box(345, 35, 115, 46, "Structured answer", "answer + evidence", green)
    arrow(150, 58, 190, 58)
    arrow(300, 58, 345, 58)
    d.add(Line(414, 120, 414, 96, strokeColor=navy, strokeWidth=1.2))
    d.add(Line(414, 96, 245, 96, strokeColor=navy, strokeWidth=1.2))
    d.add(Line(245, 96, 245, 81, strokeColor=navy, strokeWidth=1.2))
    d.add(Polygon([245, 81, 242, 87, 248, 87], fillColor=navy, strokeColor=navy))
    d.add(String(235, 176, "SYSTEM UNDER IMPLEMENTATION", textAnchor="middle", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.HexColor("#9C6500")))
    return d


def pending_figure(title: str, styles) -> KeepTogether:
    d = Drawing(470, 92)
    d.add(Rect(0, 0, 470, 92, 4, 4, fillColor=colors.HexColor("#F2F2F2"), strokeColor=colors.HexColor("#7F7F7F"), strokeWidth=1))
    d.add(String(235, 61, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=10, fillColor=colors.HexColor("#404040")))
    d.add(String(
        235,
        33,
        "Pending real-data experiment \u2014 no result available in this preliminary revision.",
        textAnchor="middle",
        fontName="Helvetica-Bold",
        fontSize=8,
        fillColor=colors.HexColor("#9C0006"),
    ))
    return KeepTogether([d, Spacer(1, 6)])


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=25, leading=29, textColor=colors.HexColor("#17365D"), alignment=TA_CENTER, spaceAfter=18),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=colors.HexColor("#17365D"), spaceAfter=9),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=colors.HexColor("#2F5597"), spaceBefore=6, spaceAfter=4),
        "h3": ParagraphStyle("H3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=colors.HexColor("#404040"), spaceBefore=5, spaceAfter=3),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=8.8, leading=11.3, alignment=TA_JUSTIFY, spaceAfter=5),
        "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=8.6, leading=10.8, leftIndent=14, firstLineIndent=-7, spaceAfter=2),
        "notice": ParagraphStyle("Notice", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=9.2, leading=12, textColor=colors.HexColor("#9C0006"), backColor=colors.HexColor("#FCE4D6"), borderColor=colors.HexColor("#C65911"), borderWidth=0.7, borderPadding=8, spaceBefore=6, spaceAfter=12),
        "code": ParagraphStyle("Code", parent=base["Code"], fontName="Courier", fontSize=7.6, leading=9.5, leftIndent=8, rightIndent=8, backColor=colors.HexColor("#F5F5F5"), borderPadding=6, spaceBefore=4, spaceAfter=7),
        "caption": ParagraphStyle("Caption", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=7.8, leading=9.5, alignment=TA_LEFT, textColor=colors.HexColor("#555555"), spaceAfter=6),
    }


def parse_markdown(text: str, styles):
    story = []
    lines = text.splitlines()
    paragraph = []
    in_code = False
    code_lines = []
    i = 0

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            raw = " ".join(x.strip() for x in paragraph)
            style = styles["notice"] if raw.startswith("&gt; PRELIMINARY") else styles["body"]
            raw = raw[5:] if raw.startswith("&gt; ") else raw
            story.append(Paragraph(inline_markup(raw), style))
            paragraph = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush_paragraph()
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue
        if line == "---PAGE---":
            flush_paragraph()
            story.append(PageBreak())
        elif line == "[ARCHITECTURE]":
            flush_paragraph()
            story.extend([architecture_diagram(), Spacer(1, 4)])
        elif line.startswith("[PENDING_FIGURE:"):
            flush_paragraph()
            title = line[len("[PENDING_FIGURE:"):-1].strip()
            story.append(pending_figure(title, styles))
        elif line.startswith("|" ):
            flush_paragraph()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", c or "-") for c in cells):
                    rows.append([Paragraph(inline_markup(c), styles["body"]) for c in cells])
                i += 1
            ncols = len(rows[0]) if rows else 1
            widths = [(PAGE_WIDTH - 1.3 * inch) / ncols] * ncols
            table = Table(rows, colWidths=widths, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#A6A6A6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([table, Spacer(1, 6)])
            continue
        elif line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(line[4:]), styles["h3"]))
        elif line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(line[3:]), styles["h2"]))
        elif line.startswith("# "):
            flush_paragraph()
            style = styles["title"] if not story else styles["h1"]
            story.append(Paragraph(inline_markup(line[2:]), style))
        elif re.match(r"^\d+\. ", line):
            flush_paragraph()
            story.append(Paragraph(inline_markup(line), styles["bullet"]))
        elif line.startswith("- "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(line[2:]), styles["bullet"], bulletText="-"))
        elif not line.strip():
            flush_paragraph()
        else:
            paragraph.append(inline_markup(line) if line.startswith("> ") else line)
        i += 1
    flush_paragraph()
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#B4C6E7"))
    canvas.line(0.65 * inch, 0.52 * inch, PAGE_WIDTH - 0.65 * inch, 0.52 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(0.65 * inch, 0.34 * inch, "CS60055 Hackathon Challenge 1 - Preliminary report")
    canvas.drawRightString(PAGE_WIDTH - 0.65 * inch, 0.34 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main() -> int:
    styles = build_styles()
    story = parse_markdown(SOURCE.read_text(encoding="utf-8"), styles)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.68 * inch,
        title="Ask the Sensors - Preliminary Technical Report",
        author="Pritam Mondal and team",
        subject="CS60055 Ubiquitous Computing Hackathon Challenge 1",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
