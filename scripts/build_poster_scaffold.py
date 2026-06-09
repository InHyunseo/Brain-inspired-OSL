#!/usr/bin/env python3
"""Build an A0 portrait PowerPoint poster scaffold.

The environment does not require python-pptx.  This script writes a minimal OOXML
.pptx package directly and also renders a PNG preview with Pillow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("output/poster")
PPTX_PATH = OUT_DIR / "osl_2d_3d_poster_scaffold.pptx"
PREVIEW_PATH = OUT_DIR / "osl_2d_3d_poster_scaffold_preview.png"

SLIDE_W_IN = 33.11
SLIDE_H_IN = 46.81
EMU_PER_IN = 914400
SLIDE_CX = int(SLIDE_W_IN * EMU_PER_IN)
SLIDE_CY = int(SLIDE_H_IN * EMU_PER_IN)

WHITE = "FFFFFF"
BLACK = "111827"
NAVY = "143A7B"
NAVY_DARK = "0F255A"
MID_BLUE = "DDE8F7"
LIGHT_BLUE = "EEF5FC"
TRACK_A = "1687A7"
TRACK_A_LIGHT = "E8F6F8"
TRACK_B = "4055A8"
TRACK_B_LIGHT = "EEF0FB"
TEAL = "159895"
ORANGE = "D9772B"
GREY = "E5E7EB"
GREY_2 = "F3F4F6"
LINE = "1F2937"


@dataclass
class Shape:
    kind: str
    x: float
    y: float
    w: float
    h: float
    fill: str | None = None
    line: str | None = None
    line_width: float = 1.0
    dash: bool = False
    text: str = ""
    font_size: float = 12.0
    bold: bool = False
    color: str = BLACK
    align: str = "l"
    valign: str = "t"
    name: str = "shape"
    margin: float = 0.08


def emu(value_in: float) -> int:
    return int(round(value_in * EMU_PER_IN))


def rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.strip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def add_rect(
    shapes: list[Shape],
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str | None,
    line: str | None = None,
    line_width: float = 1.0,
    dash: bool = False,
    name: str = "rect",
) -> None:
    shapes.append(
        Shape(
            kind="rect",
            x=x,
            y=y,
            w=w,
            h=h,
            fill=fill,
            line=line,
            line_width=line_width,
            dash=dash,
            name=name,
        )
    )


def add_text(
    shapes: list[Shape],
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    font_size: float,
    color: str = BLACK,
    bold: bool = False,
    align: str = "l",
    valign: str = "t",
    name: str = "text",
    fill: str | None = None,
    line: str | None = None,
    margin: float = 0.08,
) -> None:
    shapes.append(
        Shape(
            kind="text",
            x=x,
            y=y,
            w=w,
            h=h,
            fill=fill,
            line=line,
            text=text,
            font_size=font_size,
            color=color,
            bold=bold,
            align=align,
            valign=valign,
            name=name,
            margin=margin,
        )
    )


def add_card(
    shapes: list[Shape],
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    accent: str,
    body: str = "",
    body_size: float = 13.2,
    title_size: float = 15.0,
) -> tuple[float, float, float, float]:
    add_rect(shapes, x, y, w, h, WHITE, LINE, 1.2, name=f"{title} card")
    add_rect(shapes, x, y, w, 0.48, accent, accent, 1.0, name=f"{title} header")
    add_text(
        shapes,
        x + 0.14,
        y + 0.05,
        w - 0.28,
        0.36,
        title,
        title_size,
        WHITE,
        True,
        "l",
        "ctr",
        name=f"{title} title",
        margin=0.02,
    )
    if body:
        add_text(
            shapes,
            x + 0.22,
            y + 0.62,
            w - 0.44,
            h - 0.78,
            body,
            body_size,
            BLACK,
            False,
            "l",
            "t",
            name=f"{title} body",
            margin=0.04,
        )
    return x + 0.22, y + 0.62, w - 0.44, h - 0.78


def add_placeholder(
    shapes: list[Shape],
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    subtitle: str,
    accent: str = NAVY,
) -> None:
    add_rect(shapes, x, y, w, h, GREY_2, accent, 1.0, dash=True, name=label)
    add_text(
        shapes,
        x + 0.08,
        y + h * 0.36,
        w - 0.16,
        h * 0.30,
        f"{label}\n{subtitle}",
        13.2,
        "4B5563",
        True,
        "ctr",
        "ctr",
        name=f"{label} label",
        margin=0.02,
    )


def build_shapes() -> list[Shape]:
    shapes: list[Shape] = []

    add_rect(shapes, 0, 0, SLIDE_W_IN, SLIDE_H_IN, WHITE, None)
    add_rect(shapes, 0.55, 0.35, 3.1, 1.12, WHITE, LINE, 1.0)
    add_text(shapes, 0.55, 0.56, 3.1, 0.58, "LOGO", 20, "6B7280", True, "ctr", "ctr")
    add_rect(shapes, 29.85, 0.35, 2.55, 2.55, WHITE, LINE, 1.0)
    add_text(shapes, 29.85, 1.22, 2.55, 0.42, "QR", 20, "6B7280", True, "ctr", "ctr")

    title = "Larva-Connectome vs GRU Actors\nfor Odor-Source Localization"
    subtitle = "Cross-validation across independent 2D PPO and 3D RSAC pipelines"
    takeaway = (
        "GRU learns OSL in both tracks; the connectome actor remains unstable in both, "
        "pointing to a shared capacity/trainability bottleneck rather than a single-simulator artifact."
    )
    add_text(shapes, 3.9, 0.25, 25.6, 1.25, title, 36, NAVY_DARK, True, "ctr", "ctr")
    add_text(shapes, 4.1, 1.56, 25.2, 0.38, subtitle, 16, BLACK, False, "ctr", "ctr")
    add_text(shapes, 4.0, 2.02, 25.3, 0.38, "In Hyun Seo | [teammate] | Hanyang University", 14, BLACK, False, "ctr", "ctr")
    add_text(shapes, 4.05, 2.52, 25.2, 0.58, takeaway, 13, NAVY_DARK, True, "ctr", "ctr")
    add_rect(shapes, 0.55, 3.45, 32.0, 0.06, NAVY, None)

    add_card(
        shapes,
        0.75,
        3.78,
        31.65,
        2.65,
        "COMMON QUESTION",
        NAVY,
        (
            "Can a compact larva-connectome actor match a plain GRU for odor-source localization?\n"
            "We test the same question in two independent pipelines: 2D custom PPO and 3D ROS2/Gazebo/GADEN RSAC.\n"
            "Agreement across algorithm, simulator, and observation space makes the comparison stronger."
        ),
        body_size=13.2,
        title_size=15.5,
    )
    add_placeholder(shapes, 25.45, 4.50, 6.45, 1.42, "FIGURE 0", "larva + connectome cartoon", NAVY)

    left_x, right_x = 0.75, 16.95
    col_w = 15.45
    add_rect(shapes, left_x, 6.78, col_w, 0.66, TRACK_A, TRACK_A)
    add_text(shapes, left_x + 0.18, 6.91, col_w - 0.36, 0.35, "TRACK A - 2D OSL (PPO) | In Hyun Seo", 15, WHITE, True, "l", "ctr", margin=0.02)
    add_rect(shapes, right_x, 6.78, col_w, 0.66, TRACK_B, TRACK_B)
    add_text(shapes, right_x + 0.18, 6.91, col_w - 0.36, 0.35, "TRACK B - 3D OSL (RSAC) | teammate", 15, WHITE, True, "l", "ctr", margin=0.02)
    add_rect(shapes, left_x, 7.44, col_w, 29.0, None, TRACK_A, 1.4)
    add_rect(shapes, right_x, 7.44, col_w, 29.0, None, TRACK_B, 1.4)

    a1 = add_card(
        shapes,
        1.05,
        7.78,
        7.40,
        5.95,
        "A1. Environment",
        TRACK_A,
        "80 x 120 mm arena; source at (40, 100)\nBilateral odor + self-motion history\nClean Gaussian to turbulent bump field via alpha\nIndependent body and head axes",
        body_size=13.2,
    )
    add_placeholder(shapes, a1[0], 11.22, a1[2], 2.12, "FIGURE A1", "odor field + sensor/head schematic", TRACK_A)
    add_card(
        shapes,
        8.70,
        7.78,
        7.20,
        5.95,
        "A2. Obs / Action",
        TRACK_A,
        "Obs (6-D):\n[c_left, c_right, dlog, prev_v, prev_body_w, prev_head_w]\n\nAct (3-D):\n[v, body_w, head_w]\n\nReward: sparse goal + metabolic motion cost + dlog(c)/dt shaping",
        body_size=12.8,
    )
    add_card(
        shapes,
        1.05,
        14.10,
        14.85,
        5.22,
        "A3. Method - PPO + Selectable Actor/Critic",
        TRACK_A,
        "Custom PPO with 16 vectorized envs and staged turbulence curriculum.\nBackbone: connectome graph (423 nodes, 6 message-passing steps, D=8) or GRU hidden=421.\nCritic: default stateless MLP (64,64); recurrent critic selectable for ablations.",
        body_size=13.2,
    )
    add_placeholder(shapes, 10.55, 15.05, 5.0, 3.85, "FIGURE A3", "architecture + curriculum", TRACK_A)
    add_card(
        shapes,
        1.05,
        19.70,
        14.85,
        10.60,
        "A4. Results - Metrics vs alpha",
        TRACK_A,
        "Main readout: GRU learns clean fields and degrades gracefully under turbulence.\nActive-sensing ratio rises with turbulence and matches the hand-baseline trend.",
        body_size=13.2,
    )
    add_placeholder(shapes, 1.35, 22.15, 6.75, 3.30, "FIGURE A4a", "success + steps vs alpha", TRACK_A)
    add_placeholder(shapes, 8.40, 22.15, 7.15, 3.30, "FIGURE A4b", "active-sensing ratio vs alpha", TRACK_A)
    add_placeholder(shapes, 1.35, 25.80, 14.20, 3.85, "FIGURE A4c", "trained multi-seed trajectory overlay", TRACK_A)
    add_card(
        shapes,
        1.05,
        30.70,
        14.85,
        5.35,
        "A5. Jacobian Eigenmode Probe",
        TRACK_A,
        "Linearize hidden dynamics per step and segment by behavior.\nActive-sensing segments carry oscillatory modes that RUN lacks.",
        body_size=13.2,
    )
    add_placeholder(shapes, 9.20, 31.55, 6.35, 3.80, "FIGURE A5", "RUN vs active-sensing spectrum", TRACK_A)

    b1 = add_card(
        shapes,
        17.25,
        7.78,
        7.35,
        5.95,
        "B1. Environment",
        TRACK_B,
        "3D gas plume in ROS2 + Gazebo + GADEN\nSource at (1.0, 3.0, 0.7)\nDoor/outlet geometry; +x wind with y fluctuation",
        body_size=13.2,
    )
    add_placeholder(shapes, b1[0], 11.22, b1[2], 2.12, "FIGURE B1", "3D plume + arena render", TRACK_B)
    add_card(
        shapes,
        24.85,
        7.78,
        7.25,
        5.95,
        "B2. Observation",
        TRACK_B,
        "Richer observation than 2D:\nodor concentration + robot pose + wind + detection flag.\n\nTask: localize source within goal radius.\nMax 300 steps x 0.5 s.",
        body_size=13.0,
    )
    add_card(
        shapes,
        17.25,
        14.10,
        14.85,
        5.22,
        "B3. Method - RSAC + Two Actors",
        TRACK_B,
        "Off-policy recurrent SAC.\nGRU actor receives full observation and integrates time through hidden state.\nConnectome actor uses odor pathway plus context MLP for pose, wind, and detection.",
        body_size=13.2,
    )
    add_placeholder(shapes, 26.70, 15.05, 5.0, 3.85, "FIGURE B3", "GRU vs connectome+ctxMLP", TRACK_B)
    add_card(
        shapes,
        17.25,
        19.70,
        14.85,
        7.85,
        "B4. Results - GRU Stable",
        TRACK_B,
        "GRU policy learns stably: success rises, step-to-goal falls, and eval trajectory approaches the source region.",
        body_size=13.2,
    )
    add_placeholder(shapes, 17.55, 22.02, 4.45, 4.95, "B4a", "success curve", TRACK_B)
    add_placeholder(shapes, 22.35, 22.02, 4.45, 4.95, "B4b", "steps curve", TRACK_B)
    add_placeholder(shapes, 27.15, 22.02, 4.60, 4.95, "B4c", "eval trajectory", TRACK_B)
    add_card(
        shapes,
        17.25,
        27.95,
        14.85,
        8.10,
        "B5. Results - Connectome Partial / Unstable",
        TRACK_B,
        "Connectome policy learns only partially and remains unstable.\nFailed trajectories often drift downstream away from the source.",
        body_size=13.2,
    )
    add_placeholder(shapes, 17.55, 30.35, 4.45, 5.05, "B5a", "lower/noisier success", TRACK_B)
    add_placeholder(shapes, 22.35, 30.35, 4.45, 5.05, "B5b", "steps curve", TRACK_B)
    add_placeholder(shapes, 27.15, 30.35, 4.60, 5.05, "B5c", "downstream drift trajectory", TRACK_B)

    add_card(
        shapes,
        0.75,
        36.95,
        31.65,
        6.25,
        "SHARED CONCLUSION",
        NAVY,
        (
            "Same result, two independent pipelines: GRU actors learn odor-source localization in both 2D and 3D.\n"
            "The larva-connectome actor underperforms in both, suggesting a shared backbone-level capacity/trainability bottleneck.\n"
            "2D adds: active sensing emerges and scales with turbulence; Jacobian analysis ties it to oscillatory hidden-state dynamics.\n"
            "Future: stronger context integration, stable temporal memory, reward design, shuffled-edge/scaled-connectome controls, causal ablation."
        ),
        body_size=13.6,
        title_size=16.5,
    )
    add_rect(shapes, 1.15, 41.80, 30.85, 0.04, MID_BLUE, None)
    add_text(shapes, 0.90, 43.55, 31.30, 1.05, "Key refs: larva chemotaxis / active sensing; larva connectome; PPO + GAE; SAC | Repo: github.com/... | Contact: inhsroy@hanyang.ac.kr | Lab/funding: fill", 10.8, "374151", False, "ctr", "ctr")
    add_text(shapes, 0.90, 44.65, 31.30, 0.55, "Placeholder-first scaffold based on poster_example.pdf style: replace gray figure slots with final PNG/PDF assets.", 9.8, "6B7280", False, "ctr", "ctr")

    return shapes


def fill_xml(color: str | None) -> str:
    if color is None:
        return "<a:noFill/>"
    return f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


def line_xml(color: str | None, width: float = 1.0, dash: bool = False) -> str:
    if color is None:
        return '<a:ln><a:noFill/></a:ln>'
    dash_xml = '<a:prstDash val="dash"/>' if dash else ""
    return (
        f'<a:ln w="{int(width * 12700)}">'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        f"{dash_xml}</a:ln>"
    )


def text_body(shape: Shape) -> str:
    anchor = {"t": "t", "ctr": "ctr", "b": "b"}.get(shape.valign, "t")
    align = {"l": "l", "ctr": "ctr", "r": "r"}.get(shape.align, "l")
    inset = emu(shape.margin)
    runs = []
    for raw_line in shape.text.splitlines() or [""]:
        line = escape(raw_line)
        bold = ' b="1"' if shape.bold else ""
        runs.append(
            "<a:p>"
            f'<a:pPr algn="{align}"/>'
            "<a:r>"
            f'<a:rPr lang="en-US" sz="{int(shape.font_size * 100)}"{bold}>'
            f'<a:solidFill><a:srgbClr val="{shape.color}"/></a:solidFill>'
            '<a:latin typeface="Arial"/>'
            "</a:rPr>"
            f"<a:t>{line}</a:t>"
            "</a:r>"
            f'<a:endParaRPr lang="en-US" sz="{int(shape.font_size * 100)}"/>'
            "</a:p>"
        )
    return (
        f'<p:txBody><a:bodyPr wrap="square" anchor="{anchor}" '
        f'lIns="{inset}" tIns="{inset}" rIns="{inset}" bIns="{inset}">'
        "<a:noAutofit/></a:bodyPr><a:lstStyle/>"
        + "".join(runs)
        + "</p:txBody>"
    )


def shape_xml(shape: Shape, shape_id: int) -> str:
    tx_body = text_body(shape) if shape.text else "<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>"
    return (
        "<p:sp>"
        "<p:nvSpPr>"
        f'<p:cNvPr id="{shape_id}" name="{escape(shape.name)}"/>'
        '<p:cNvSpPr txBox="1"/>'
        "<p:nvPr/>"
        "</p:nvSpPr>"
        "<p:spPr>"
        "<a:xfrm>"
        f'<a:off x="{emu(shape.x)}" y="{emu(shape.y)}"/>'
        f'<a:ext cx="{emu(shape.w)}" cy="{emu(shape.h)}"/>'
        "</a:xfrm>"
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f"{fill_xml(shape.fill)}"
        f"{line_xml(shape.line, shape.line_width, shape.dash)}"
        "</p:spPr>"
        f"{tx_body}"
        "</p:sp>"
    )


def slide_xml(shapes: Iterable[Shape]) -> str:
    body = "\n".join(shape_xml(shape, idx + 2) for idx, shape in enumerate(shapes))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="{WHITE}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_CX}" cy="{SLIDE_CY}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_CX}" cy="{SLIDE_CY}"/></a:xfrm></p:grpSpPr>
      {body}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
'''


def minimal_theme_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme">
  <a:themeElements>
    <a:clrScheme name="Office">
      <a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F3F4F6"/></a:lt2>
      <a:accent1><a:srgbClr val="143A7B"/></a:accent1><a:accent2><a:srgbClr val="1687A7"/></a:accent2>
      <a:accent3><a:srgbClr val="D9772B"/></a:accent3><a:accent4><a:srgbClr val="4055A8"/></a:accent4>
      <a:accent5><a:srgbClr val="159895"/></a:accent5><a:accent6><a:srgbClr val="6B7280"/></a:accent6>
      <a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Office"><a:majorFont><a:latin typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Office">
      <a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
      <a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/><a:extraClrSchemeLst/>
</a:theme>
'''


def write_pptx(shapes: list[Shape]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    files = {
        "[Content_Types].xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>
  <Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>
  <Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
''',
        "_rels/.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
''',
        "docProps/core.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>OSL 2D 3D Poster Scaffold</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
''',
        "docProps/app.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex OOXML builder</Application><PresentationFormat>A0 Portrait</PresentationFormat><Slides>1</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides>
</Properties>
''',
        "ppt/presentation.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
  <p:sldSz cx="{SLIDE_CX}" cy="{SLIDE_CY}" type="custom"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
''',
        "ppt/_rels/presentation.xml.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>
''',
        "ppt/slides/slide1.xml": slide_xml(shapes),
        "ppt/slides/_rels/slide1.xml.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>
''',
        "ppt/slideMasters/slideMaster1.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>
''',
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
''',
        "ppt/slideLayouts/slideLayout1.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
''',
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
''',
        "ppt/theme/theme1.xml": minimal_theme_xml(),
        "ppt/presProps.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentationPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>''',
        "ppt/viewProps.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:viewPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>''',
        "ppt/tableStyles.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>''',
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(PPTX_PATH, "w", ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_lines(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    wrapped: list[str] = []
    for para in text.splitlines():
        words = para.split()
        if not words:
            wrapped.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                line = candidate
            else:
                wrapped.append(line)
                line = word
        wrapped.append(line)
    return wrapped


def draw_dashed_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int], width: int) -> None:
    x0, y0, x1, y1 = box
    dash, gap = max(6, width * 4), max(4, width * 3)
    for x in range(x0, x1, dash + gap):
        draw.line((x, y0, min(x + dash, x1), y0), fill=color, width=width)
        draw.line((x, y1, min(x + dash, x1), y1), fill=color, width=width)
    for y in range(y0, y1, dash + gap):
        draw.line((x0, y, x0, min(y + dash, y1)), fill=color, width=width)
        draw.line((x1, y, x1, min(y + dash, y1)), fill=color, width=width)


def render_preview(shapes: list[Shape]) -> None:
    scale = 72
    img = Image.new("RGB", (int(SLIDE_W_IN * scale), int(SLIDE_H_IN * scale)), rgb(WHITE))
    draw = ImageDraw.Draw(img)
    for shape in shapes:
        x0, y0 = int(shape.x * scale), int(shape.y * scale)
        x1, y1 = int((shape.x + shape.w) * scale), int((shape.y + shape.h) * scale)
        if shape.fill:
            draw.rectangle((x0, y0, x1, y1), fill=rgb(shape.fill))
        if shape.line:
            lw = max(1, int(shape.line_width * scale / 36))
            if shape.dash:
                draw_dashed_rect(draw, (x0, y0, x1, y1), rgb(shape.line), lw)
            else:
                draw.rectangle((x0, y0, x1, y1), outline=rgb(shape.line), width=lw)
        if shape.text:
            font = load_font(max(7, int(shape.font_size * scale / 72)), shape.bold)
            margin = int(shape.margin * scale)
            max_width = max(10, (x1 - x0) - 2 * margin)
            lines = wrap_lines(shape.text, font, max_width, draw)
            line_h = int(font.size * 1.18) if hasattr(font, "size") else 14
            total_h = line_h * len(lines)
            if shape.valign == "ctr":
                ty = y0 + max(margin, ((y1 - y0) - total_h) // 2)
            elif shape.valign == "b":
                ty = y1 - total_h - margin
            else:
                ty = y0 + margin
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                if shape.align == "ctr":
                    tx = x0 + ((x1 - x0) - (bbox[2] - bbox[0])) // 2
                elif shape.align == "r":
                    tx = x1 - margin - (bbox[2] - bbox[0])
                else:
                    tx = x0 + margin
                draw.text((tx, ty), line, font=font, fill=rgb(shape.color))
                ty += line_h
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(PREVIEW_PATH)


def main() -> None:
    shapes = build_shapes()
    write_pptx(shapes)
    render_preview(shapes)
    print(PPTX_PATH)
    print(PREVIEW_PATH)


if __name__ == "__main__":
    main()
