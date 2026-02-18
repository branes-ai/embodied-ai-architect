#!/usr/bin/env python3
"""Render pitch deck slides as PNG images using Pillow (no LibreOffice needed)."""

from PIL import Image, ImageDraw, ImageFont
import textwrap

W, H = 1920, 1080
BLACK = (0x1A, 0x1A, 0x2E)
DARK = (0x22, 0x2B, 0x45)
WHITE = (0xFF, 0xFF, 0xFF)
LIGHT_GRAY = (0xCC, 0xCC, 0xCC)
ACCENT = (0xE8, 0x4D, 0x3D)
ACCENT2 = (0x3D, 0x9B, 0xE9)
MUTED = (0x88, 0x88, 0x99)
GREEN = (0x4E, 0xC9, 0xB0)

# Try to load a good font, fall back to default
try:
    FONT_B_44 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44)
    FONT_B_28 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    FONT_B_22 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    FONT_B_18 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    FONT_22 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    FONT_20 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    FONT_18 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    FONT_16 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except:
    FONT_B_44 = ImageFont.load_default()
    FONT_B_28 = FONT_B_22 = FONT_B_18 = FONT_B_44
    FONT_22 = FONT_20 = FONT_18 = FONT_16 = FONT_B_44


def new_slide():
    img = Image.new("RGB", (W, H), BLACK)
    return img, ImageDraw.Draw(img)


def accent_line(draw, x, y, w=150):
    draw.rectangle([x, y, x + w, y + 3], fill=ACCENT)


def draw_rounded_rect(draw, xy, fill, r=12):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=r, fill=fill)


# ---- SLIDE 1 ----
img1, d = new_slide()
d.text((80, 90), "YOUR AUTONOMY RUNS ON", font=FONT_B_44, fill=WHITE)
d.text((80, 145), "SOMEONE ELSE'S ROADMAP", font=FONT_B_44, fill=WHITE)
accent_line(d, 80, 220)
bullets = [
    "Their highest-volume market isn't yours",
    "Allocation shortages strand programs overnight",
    "Export controls redraw the map without warning",
]
for i, b in enumerate(bullets):
    d.text((80, 270 + i * 55), f"\u2014  {b}", font=FONT_22, fill=LIGHT_GRAY)
d.text((80, 480), "If you don't own the silicon strategy, you don't own the autonomy.",
       font=FONT_20, fill=ACCENT)
img1.save("docs/slide1.png")

# ---- SLIDE 2 ----
img2, d = new_slide()
d.text((80, 70), "THE GAP IS GETTING WORSE", font=FONT_B_44, fill=WHITE)
accent_line(d, 80, 140)

cols = [
    ("PHYSICS", ACCENT, "5W drone\n30W robot", "Workloads growing\nScaling slowing"),
    ("GEOPOLITICS", ACCENT2, "2 countries fab\nadvanced silicon", "Export controls\nexpanding yearly"),
    ("ECONOMICS", GREEN, "$50 chip creates\n$500 system penalty", "Competitors won't\noverpay forever"),
]
for i, (title, color, constraint, trend) in enumerate(cols):
    x = 80 + i * 600
    d.text((x, 200), title, font=FONT_B_22, fill=color)
    for j, line in enumerate(constraint.split("\n")):
        d.text((x, 260 + j * 30), line, font=FONT_20, fill=WHITE)
    for j, line in enumerate(trend.split("\n")):
        d.text((x, 360 + j * 28), line, font=FONT_18, fill=MUTED)
img2.save("docs/slide2.png")

# ---- SLIDE 3 ----
img3, d = new_slide()
d.text((80, 70), "DESCRIBE THE MISSION.", font=FONT_B_44, fill=WHITE)
d.text((80, 125), "GET THE SILICON.", font=FONT_B_44, fill=WHITE)
accent_line(d, 80, 200)

# Input box
draw_rounded_rect(d, (80, 270, 700, 430), fill=DARK)
d.text((100, 280), "INPUT", font=FONT_B_18, fill=MUTED)
d.text((100, 315), '"Delivery drone, visual SLAM +', font=FONT_20, fill=LIGHT_GRAY)
d.text((100, 345), ' detection, 30fps, under 5 watts"', font=FONT_20, fill=LIGHT_GRAY)

# Arrow
d.text((770, 320), "\u25B6", font=FONT_B_44, fill=ACCENT)

# Output box
draw_rounded_rect(d, (900, 270, 1600, 430), fill=DARK)
d.text((920, 280), "OUTPUT", font=FONT_B_18, fill=MUTED)
d.text((920, 315), "Validated SoC architecture", font=FONT_B_22, fill=WHITE)
d.text((920, 350), "with synthesizable Verilog", font=FONT_B_22, fill=WHITE)

# Pipeline steps
steps = ["Workload Analysis", "Architecture Exploration", "Constraint Validation", "RTL Generation"]
for i, step in enumerate(steps):
    x = 80 + i * 440
    d.text((x, 490), step, font=FONT_18, fill=ACCENT2)
    if i < len(steps) - 1:
        d.text((x + 300, 490), "\u2192", font=FONT_20, fill=MUTED)

d.text((80, 560), "One engineer.  One afternoon.", font=FONT_B_22, fill=ACCENT)
img3.save("docs/slide3.png")

# ---- SLIDE 4 ----
img4, d = new_slide()
d.text((80, 70), "IT WORKS TODAY", font=FONT_B_44, fill=WHITE)
accent_line(d, 80, 140)

proofs = [
    ("DRONE SoC", "Auto-optimized to\nall-PASS constraints", "1.5 sec"),
    ("QUADRUPED SoC", "4 concurrent workloads\nPareto-ranked", "< 2 sec"),
    ("RTL PIPELINE", "Synthesizable Verilog\nverified with Yosys", "seconds"),
]
for i, (title, desc, time) in enumerate(proofs):
    x = 80 + i * 580
    draw_rounded_rect(d, (x, 200, x + 520, 480), fill=DARK)
    d.text((x + 25, 220), title, font=FONT_B_22, fill=ACCENT)
    for j, line in enumerate(desc.split("\n")):
        d.text((x + 25, 275 + j * 30), line, font=FONT_20, fill=WHITE)
    d.text((x + 25, 390), time, font=FONT_B_28, fill=ACCENT2)

d.text((80, 520), "50+ COTS platforms profiled  \u00B7  Custom accelerator RTL  \u00B7  No cloud dependency",
       font=FONT_18, fill=MUTED)
img4.save("docs/slide4.png")

# ---- SLIDE 5 ----
img5, d = new_slide()
d.text((80, 70), "OWN THE COMPUTE.", font=FONT_B_44, fill=WHITE)
d.text((80, 125), "OWN THE AUTONOMY.", font=FONT_B_44, fill=WHITE)
accent_line(d, 80, 200)

phases = [
    ("EVALUATE", "2 weeks", "Your workloads on\nCOTS + custom targets"),
    ("DESIGN", "8 weeks", "Validated SoC\narchitecture with RTL"),
    ("VERIFY", "12 weeks", "Pre-silicon validation\ntape-out ready"),
]
for i, (phase, timeline, desc) in enumerate(phases):
    x = 80 + i * 580
    draw_rounded_rect(d, (x, 260, x + 520, 500), fill=DARK)
    d.text((x + 25, 280), phase, font=FONT_B_22, fill=ACCENT2)
    d.text((x + 25, 325), timeline, font=FONT_B_28, fill=WHITE)
    for j, line in enumerate(desc.split("\n")):
        d.text((x + 25, 390 + j * 28), line, font=FONT_18, fill=LIGHT_GRAY)
    # Arrow between boxes
    if i < len(phases) - 1:
        d.text((x + 540, 350), "\u25B6", font=FONT_B_28, fill=MUTED)

d.text((80, 550), "Next step:  2-week evaluation sprint.", font=FONT_B_22, fill=WHITE)
d.text((80, 590), "You bring models.  We bring the platform.", font=FONT_B_22, fill=WHITE)
img5.save("docs/slide5.png")

print("Rendered: docs/slide1.png through docs/slide5.png")
