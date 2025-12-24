#!/usr/bin/env python3
"""
SoC Optimization Workflow - LangGraph-based optimization loop.

This script demonstrates the LangGraph SoC optimizer using a RISC-V ALU
as the test design. It runs the optimization loop and reports results.

Usage:
    python workflow.py                    # Run with mock mode (no LLM)
    python workflow.py --with-llm         # Run with actual LLM calls
    python workflow.py --simple           # Run simple loop (no LangGraph)
    python workflow.py --help             # Show options

Requirements:
    - Yosys (for synthesis)
    - Verilator or Icarus Verilog (for simulation)
    - Optional: langgraph (pip install langgraph)
    - Optional: langchain-anthropic (for LLM integration)
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from state import DesignConstraints, state_summary


# =============================================================================
# TECHNOLOGY CONFIGURATION
# =============================================================================
# Reference areas for common components at different process nodes.
# These are approximate values for architectural planning based on:
# - ITRS/IRDS roadmaps
# - Published literature
# - Open-source PDK data (SKY130, GF180)
#
# Note: For actual synthesis, only open-source PDKs are available:
#   - SKY130 (130nm) - fully supported by Yosys/OpenROAD
#   - GF180MCU (180nm) - fully supported
#   - IHP SG13G2 (130nm BiCMOS) - supported
# Advanced nodes require proprietary PDKs with foundry NDAs.
# =============================================================================

TECHNOLOGY_NODES = {
    # =========================================================================
    # GAA / Nanosheet Era (2nm - 3nm)
    # =========================================================================
    "2nm": {
        "name": "2nm GAA (N2)",
        "vendor": "TSMC/Intel/Samsung",
        "generation": "GAA Nanosheet",
        "year": 2025,
        "cell_area_um2": 0.020,
        "gate_delay_ps": 5,
        "energy_per_cell_pj": 0.003,
        "vdd_v": 0.55,
        "reference_areas": {
            "INV_X1": 0.008, "NAND2_X1": 0.012, "NOR2_X1": 0.014, "DFF_X1": 0.080,
            "REG_32bit": 2.6, "ADDER_32bit": 6.0, "ADDER_32bit_CLA": 10.0,
            "MUL_32bit": 320.0, "MUL_32bit_BOOTH": 200.0,
            "SRAM_1KB": 100.0, "SRAM_32KB": 3200.0,
            "ALU_32bit": 40.0, "FPU_32bit": 800.0,
        },
    },
    "3nm": {
        "name": "3nm GAA (N3)",
        "vendor": "TSMC/Samsung",
        "generation": "GAA Nanosheet",
        "year": 2022,
        "cell_area_um2": 0.025,
        "gate_delay_ps": 6,
        "energy_per_cell_pj": 0.004,
        "vdd_v": 0.60,
        "reference_areas": {
            "INV_X1": 0.010, "NAND2_X1": 0.015, "NOR2_X1": 0.018, "DFF_X1": 0.100,
            "REG_32bit": 3.2, "ADDER_32bit": 7.5, "ADDER_32bit_CLA": 12.5,
            "MUL_32bit": 400.0, "MUL_32bit_BOOTH": 250.0,
            "SRAM_1KB": 125.0, "SRAM_32KB": 4000.0,
            "ALU_32bit": 50.0, "FPU_32bit": 1000.0,
        },
    },
    # =========================================================================
    # EUV FinFET Era (4nm - 5nm)
    # =========================================================================
    "4nm": {
        "name": "4nm EUV FinFET (N4)",
        "vendor": "TSMC/Samsung",
        "generation": "EUV FinFET",
        "year": 2021,
        "cell_area_um2": 0.030,
        "gate_delay_ps": 7,
        "energy_per_cell_pj": 0.005,
        "vdd_v": 0.65,
        "reference_areas": {
            "INV_X1": 0.012, "NAND2_X1": 0.018, "NOR2_X1": 0.021, "DFF_X1": 0.120,
            "REG_32bit": 3.8, "ADDER_32bit": 9.0, "ADDER_32bit_CLA": 15.0,
            "MUL_32bit": 480.0, "MUL_32bit_BOOTH": 300.0,
            "SRAM_1KB": 150.0, "SRAM_32KB": 4800.0,
            "ALU_32bit": 60.0, "FPU_32bit": 1200.0,
        },
    },
    "5nm": {
        "name": "5nm EUV FinFET (N5)",
        "vendor": "TSMC/Samsung",
        "generation": "EUV FinFET",
        "year": 2020,
        "cell_area_um2": 0.040,
        "gate_delay_ps": 8,
        "energy_per_cell_pj": 0.007,
        "vdd_v": 0.65,
        "reference_areas": {
            "INV_X1": 0.016, "NAND2_X1": 0.024, "NOR2_X1": 0.028, "DFF_X1": 0.160,
            "REG_32bit": 5.1, "ADDER_32bit": 12.0, "ADDER_32bit_CLA": 20.0,
            "MUL_32bit": 640.0, "MUL_32bit_BOOTH": 400.0,
            "SRAM_1KB": 200.0, "SRAM_32KB": 6400.0,
            "ALU_32bit": 80.0, "FPU_32bit": 1600.0,
        },
    },
    # =========================================================================
    # Advanced FinFET Era (7nm - 8nm)
    # =========================================================================
    "7nm": {
        "name": "7nm FinFET (N7)",
        "vendor": "TSMC/Samsung",
        "generation": "DUV/EUV FinFET",
        "year": 2018,
        "cell_area_um2": 0.050,
        "gate_delay_ps": 10,
        "energy_per_cell_pj": 0.010,
        "vdd_v": 0.70,
        "reference_areas": {
            "INV_X1": 0.020, "NAND2_X1": 0.030, "NOR2_X1": 0.035, "DFF_X1": 0.200,
            "REG_32bit": 6.4, "ADDER_32bit": 15.0, "ADDER_32bit_CLA": 25.0,
            "MUL_32bit": 800.0, "MUL_32bit_BOOTH": 500.0,
            "SRAM_1KB": 250.0, "SRAM_32KB": 8000.0,
            "ALU_32bit": 100.0, "FPU_32bit": 2000.0,
        },
    },
    "8nm": {
        "name": "8nm FinFET (N8)",
        "vendor": "Samsung",
        "generation": "DUV FinFET",
        "year": 2018,
        "cell_area_um2": 0.060,
        "gate_delay_ps": 11,
        "energy_per_cell_pj": 0.012,
        "vdd_v": 0.70,
        "reference_areas": {
            "INV_X1": 0.024, "NAND2_X1": 0.036, "NOR2_X1": 0.042, "DFF_X1": 0.240,
            "REG_32bit": 7.7, "ADDER_32bit": 18.0, "ADDER_32bit_CLA": 30.0,
            "MUL_32bit": 960.0, "MUL_32bit_BOOTH": 600.0,
            "SRAM_1KB": 300.0, "SRAM_32KB": 9600.0,
            "ALU_32bit": 120.0, "FPU_32bit": 2400.0,
        },
    },
    # =========================================================================
    # First-Gen FinFET Era (10nm - 16nm)
    # =========================================================================
    "10nm": {
        "name": "10nm FinFET (N10)",
        "vendor": "TSMC/Samsung/Intel",
        "generation": "FinFET",
        "year": 2016,
        "cell_area_um2": 0.070,
        "gate_delay_ps": 12,
        "energy_per_cell_pj": 0.015,
        "vdd_v": 0.75,
        "reference_areas": {
            "INV_X1": 0.028, "NAND2_X1": 0.042, "NOR2_X1": 0.049, "DFF_X1": 0.280,
            "REG_32bit": 9.0, "ADDER_32bit": 21.0, "ADDER_32bit_CLA": 35.0,
            "MUL_32bit": 1120.0, "MUL_32bit_BOOTH": 700.0,
            "SRAM_1KB": 350.0, "SRAM_32KB": 11200.0,
            "ALU_32bit": 140.0, "FPU_32bit": 2800.0,
        },
    },
    "12nm": {
        "name": "12nm FinFET (N12)",
        "vendor": "TSMC/GlobalFoundries",
        "generation": "FinFET",
        "year": 2017,
        "cell_area_um2": 0.085,
        "gate_delay_ps": 13,
        "energy_per_cell_pj": 0.018,
        "vdd_v": 0.75,
        "reference_areas": {
            "INV_X1": 0.034, "NAND2_X1": 0.051, "NOR2_X1": 0.060, "DFF_X1": 0.340,
            "REG_32bit": 10.9, "ADDER_32bit": 25.5, "ADDER_32bit_CLA": 42.5,
            "MUL_32bit": 1360.0, "MUL_32bit_BOOTH": 850.0,
            "SRAM_1KB": 425.0, "SRAM_32KB": 13600.0,
            "ALU_32bit": 170.0, "FPU_32bit": 3400.0,
        },
    },
    "14nm": {
        "name": "14nm FinFET (N14)",
        "vendor": "Intel/TSMC/Samsung",
        "generation": "FinFET",
        "year": 2014,
        "cell_area_um2": 0.100,
        "gate_delay_ps": 15,
        "energy_per_cell_pj": 0.020,
        "vdd_v": 0.80,
        "reference_areas": {
            "INV_X1": 0.040, "NAND2_X1": 0.060, "NOR2_X1": 0.070, "DFF_X1": 0.400,
            "REG_32bit": 12.8, "ADDER_32bit": 30.0, "ADDER_32bit_CLA": 50.0,
            "MUL_32bit": 1600.0, "MUL_32bit_BOOTH": 1000.0,
            "SRAM_1KB": 500.0, "SRAM_32KB": 16000.0,
            "ALU_32bit": 200.0, "FPU_32bit": 4000.0,
        },
    },
    "16nm": {
        "name": "16nm FinFET (N16)",
        "vendor": "TSMC",
        "generation": "FinFET",
        "year": 2013,
        "cell_area_um2": 0.120,
        "gate_delay_ps": 17,
        "energy_per_cell_pj": 0.025,
        "vdd_v": 0.80,
        "reference_areas": {
            "INV_X1": 0.048, "NAND2_X1": 0.072, "NOR2_X1": 0.084, "DFF_X1": 0.480,
            "REG_32bit": 15.4, "ADDER_32bit": 36.0, "ADDER_32bit_CLA": 60.0,
            "MUL_32bit": 1920.0, "MUL_32bit_BOOTH": 1200.0,
            "SRAM_1KB": 600.0, "SRAM_32KB": 19200.0,
            "ALU_32bit": 240.0, "FPU_32bit": 4800.0,
        },
    },
    # =========================================================================
    # Planar CMOS Era (22nm - 28nm)
    # =========================================================================
    "22nm": {
        "name": "22nm Planar/TriGate",
        "vendor": "Intel/GlobalFoundries",
        "generation": "Planar/TriGate",
        "year": 2012,
        "cell_area_um2": 0.160,
        "gate_delay_ps": 20,
        "energy_per_cell_pj": 0.035,
        "vdd_v": 0.85,
        "reference_areas": {
            "INV_X1": 0.064, "NAND2_X1": 0.096, "NOR2_X1": 0.112, "DFF_X1": 0.640,
            "REG_32bit": 20.5, "ADDER_32bit": 48.0, "ADDER_32bit_CLA": 80.0,
            "MUL_32bit": 2560.0, "MUL_32bit_BOOTH": 1600.0,
            "SRAM_1KB": 800.0, "SRAM_32KB": 25600.0,
            "ALU_32bit": 320.0, "FPU_32bit": 6400.0,
        },
    },
    "28nm": {
        "name": "28nm Planar CMOS",
        "vendor": "TSMC/GlobalFoundries/Samsung",
        "generation": "Planar CMOS",
        "year": 2011,
        "cell_area_um2": 0.200,
        "gate_delay_ps": 25,
        "energy_per_cell_pj": 0.050,
        "vdd_v": 0.90,
        "reference_areas": {
            "INV_X1": 0.080, "NAND2_X1": 0.120, "NOR2_X1": 0.140, "DFF_X1": 0.800,
            "REG_32bit": 25.6, "ADDER_32bit": 60.0, "ADDER_32bit_CLA": 100.0,
            "MUL_32bit": 3200.0, "MUL_32bit_BOOTH": 2000.0,
            "SRAM_1KB": 1000.0, "SRAM_32KB": 32000.0,
            "ALU_32bit": 400.0, "FPU_32bit": 8000.0,
        },
    },
    # =========================================================================
    # Mature Planar CMOS (40nm - 65nm)
    # =========================================================================
    "40nm": {
        "name": "40nm Planar CMOS",
        "vendor": "TSMC/GlobalFoundries",
        "generation": "Planar CMOS",
        "year": 2008,
        "cell_area_um2": 0.350,
        "gate_delay_ps": 35,
        "energy_per_cell_pj": 0.080,
        "vdd_v": 1.00,
        "reference_areas": {
            "INV_X1": 0.140, "NAND2_X1": 0.210, "NOR2_X1": 0.245, "DFF_X1": 1.400,
            "REG_32bit": 44.8, "ADDER_32bit": 105.0, "ADDER_32bit_CLA": 175.0,
            "MUL_32bit": 5600.0, "MUL_32bit_BOOTH": 3500.0,
            "SRAM_1KB": 1750.0, "SRAM_32KB": 56000.0,
            "ALU_32bit": 700.0, "FPU_32bit": 14000.0,
        },
    },
    "65nm": {
        "name": "65nm Planar CMOS",
        "vendor": "TSMC/GlobalFoundries",
        "generation": "Planar CMOS",
        "year": 2006,
        "cell_area_um2": 0.600,
        "gate_delay_ps": 50,
        "energy_per_cell_pj": 0.120,
        "vdd_v": 1.10,
        "reference_areas": {
            "INV_X1": 0.240, "NAND2_X1": 0.360, "NOR2_X1": 0.420, "DFF_X1": 2.400,
            "REG_32bit": 76.8, "ADDER_32bit": 180.0, "ADDER_32bit_CLA": 300.0,
            "MUL_32bit": 9600.0, "MUL_32bit_BOOTH": 6000.0,
            "SRAM_1KB": 3000.0, "SRAM_32KB": 96000.0,
            "ALU_32bit": 1200.0, "FPU_32bit": 24000.0,
        },
    },
    # =========================================================================
    # Legacy Planar CMOS (90nm - 130nm) - Open PDKs Available
    # =========================================================================
    "90nm": {
        "name": "90nm Planar CMOS",
        "vendor": "TSMC/IBM",
        "generation": "Planar CMOS",
        "year": 2004,
        "cell_area_um2": 1.000,
        "gate_delay_ps": 70,
        "energy_per_cell_pj": 0.200,
        "vdd_v": 1.20,
        "reference_areas": {
            "INV_X1": 0.400, "NAND2_X1": 0.600, "NOR2_X1": 0.700, "DFF_X1": 4.000,
            "REG_32bit": 128.0, "ADDER_32bit": 300.0, "ADDER_32bit_CLA": 500.0,
            "MUL_32bit": 16000.0, "MUL_32bit_BOOTH": 10000.0,
            "SRAM_1KB": 5000.0, "SRAM_32KB": 160000.0,
            "ALU_32bit": 2000.0, "FPU_32bit": 40000.0,
        },
    },
    "130nm": {
        "name": "130nm Planar CMOS (SKY130)",
        "vendor": "SkyWater/Google (Open PDK)",
        "generation": "Planar CMOS",
        "year": 2001,
        "open_pdk": True,
        "cell_area_um2": 1.600,
        "gate_delay_ps": 100,
        "energy_per_cell_pj": 0.350,
        "vdd_v": 1.80,
        "reference_areas": {
            "INV_X1": 0.640, "NAND2_X1": 0.960, "NOR2_X1": 1.120, "DFF_X1": 6.400,
            "REG_32bit": 204.8, "ADDER_32bit": 480.0, "ADDER_32bit_CLA": 800.0,
            "MUL_32bit": 25600.0, "MUL_32bit_BOOTH": 16000.0,
            "SRAM_1KB": 8000.0, "SRAM_32KB": 256000.0,
            "ALU_32bit": 3200.0, "FPU_32bit": 64000.0,
        },
    },
    "180nm": {
        "name": "180nm Planar CMOS (GF180)",
        "vendor": "GlobalFoundries (Open PDK)",
        "generation": "Planar CMOS",
        "year": 1999,
        "open_pdk": True,
        "cell_area_um2": 3.000,
        "gate_delay_ps": 150,
        "energy_per_cell_pj": 0.600,
        "vdd_v": 1.80,
        "reference_areas": {
            "INV_X1": 1.200, "NAND2_X1": 1.800, "NOR2_X1": 2.100, "DFF_X1": 12.000,
            "REG_32bit": 384.0, "ADDER_32bit": 900.0, "ADDER_32bit_CLA": 1500.0,
            "MUL_32bit": 48000.0, "MUL_32bit_BOOTH": 30000.0,
            "SRAM_1KB": 15000.0, "SRAM_32KB": 480000.0,
            "ALU_32bit": 6000.0, "FPU_32bit": 120000.0,
        },
    },
}

DEFAULT_TECHNOLOGY = "7nm"


def load_designs():
    """Load the test RTL and testbench."""
    designs_dir = Path(__file__).parent / "designs"

    rtl_path = designs_dir / "riscv_alu.sv"
    tb_path = designs_dir / "riscv_alu_tb.sv"

    if not rtl_path.exists():
        print(f"Error: RTL file not found: {rtl_path}")
        sys.exit(1)

    rtl_code = rtl_path.read_text()
    testbench = tb_path.read_text() if tb_path.exists() else None

    return rtl_code, testbench


def run_baseline_assessment(
    rtl_code: str,
    testbench: str,
    top_module: str,
    work_dir: Path,
    target_clock_ns: float = 1.0,
    technology: str = DEFAULT_TECHNOLOGY,
    verbose: bool = True,
) -> dict:
    """
    Run baseline PPA assessment for human review.

    This is the first step in human-in-the-loop optimization.
    Reports metrics without making any optimization decisions.
    """
    from tools import RTLLintTool, RTLSynthesisTool, SimulationTool

    work_dir = Path(work_dir).resolve()
    lint_tool = RTLLintTool(work_dir / "lint")
    synth_tool = RTLSynthesisTool(work_dir / "synth")
    sim_tool = SimulationTool(work_dir / "sim")

    # Get technology parameters
    tech = TECHNOLOGY_NODES.get(technology, TECHNOLOGY_NODES[DEFAULT_TECHNOLOGY])

    result = {
        "success": False,
        "lint_passed": False,
        "synthesis_passed": False,
        "validation_passed": None,
        "metrics": {},
        "technology": tech,
    }

    # Step 1: Lint
    if verbose:
        print("\n[1/3] Syntax Check (Verilator)...")
    lint_result = lint_tool.run(rtl_code)
    result["lint_passed"] = lint_result.get("success", False)

    if not result["lint_passed"]:
        print(f"  FAILED: {lint_result.get('errors', ['Unknown error'])[:2]}")
        return result

    if verbose:
        warnings = lint_result.get("warnings", [])
        print(f"  PASSED ({len(warnings)} warnings)")

    # Step 2: Synthesis
    if verbose:
        print("\n[2/3] Synthesis (Yosys)...")
    synth_result = synth_tool.run(rtl_code, top_module)
    result["synthesis_passed"] = synth_result.get("success", False)

    if not result["synthesis_passed"]:
        print(f"  FAILED: {synth_result.get('errors', ['Unknown error'])[:2]}")
        return result

    # Extract and convert metrics using technology parameters
    area_cells = synth_result.get("area_cells", 0)
    num_wires = synth_result.get("num_wires", 0)
    cell_counts = synth_result.get("cell_counts", {})

    # Technology-dependent calculations
    cell_area_um2 = tech["cell_area_um2"]
    gate_delay_ps = tech["gate_delay_ps"]
    energy_per_cell_pj = tech["energy_per_cell_pj"]

    area_um2 = area_cells * cell_area_um2

    # Estimate timing (rough: logic depth * gate delay)
    # Estimate 20 logic levels for 32-bit ALU
    LOGIC_LEVELS = 20
    latency_ps = LOGIC_LEVELS * gate_delay_ps

    # Estimate energy per operation
    energy_pj = area_cells * energy_per_cell_pj

    result["metrics"] = {
        "area_cells": area_cells,
        "area_um2": area_um2,
        "num_wires": num_wires,
        "latency_ps": latency_ps,
        "energy_pj": energy_pj,
        "target_clock_ns": target_clock_ns,
        "target_clock_ps": target_clock_ns * 1000,
        "timing_slack_ps": (target_clock_ns * 1000) - latency_ps,
        "cell_counts": cell_counts,
    }

    if verbose:
        print(f"  PASSED")

    # Step 3: Validation (if testbench provided)
    if testbench:
        if verbose:
            print("\n[3/3] Functional Validation (Icarus Verilog)...")
        sim_result = sim_tool.run(rtl_code, testbench, top_module)
        result["validation_passed"] = sim_result.get("success", False)

        if result["validation_passed"]:
            tests_passed = sim_result.get("tests_passed", 0)
            print(f"  PASSED ({tests_passed} tests)")
        else:
            tests_failed = sim_result.get("tests_failed", 0)
            print(f"  FAILED ({tests_failed} tests failed)")
            return result
    else:
        if verbose:
            print("\n[3/3] Functional Validation: SKIPPED (no testbench)")

    result["success"] = True
    return result


def print_ppa_report(metrics: dict, module_name: str, technology: dict):
    """Print PPA assessment report for human review."""
    W = 66  # Inner width of box (between │ characters)

    def box_line(text, pad=" "):
        """Create a box line with proper padding."""
        return f"  │{pad}{text:<{W-2}}{pad}│"

    def box_top():
        return "  ┌" + "─" * W + "┐"

    def box_mid():
        return "  ├" + "─" * W + "┤"

    def box_bot():
        return "  └" + "─" * W + "┘"

    print("\n" + "=" * 70)
    print("  PPA BASELINE ASSESSMENT")
    print("=" * 70)

    # Technology header
    print(f"\n  Technology: {technology['name']}")
    print(f"  Vendor:     {technology['vendor']}")
    print(f"  Generation: {technology.get('generation', 'N/A')} ({technology.get('year', 'N/A')})")
    print(f"  Vdd:        {technology.get('vdd_v', 'N/A')} V")
    is_open = technology.get('open_pdk', False)
    print(f"  PDK:        {'Open Source (Yosys/OpenROAD)' if is_open else 'Proprietary (estimation only)'}")
    print(f"\n  Module:     {module_name}")

    # Design Metrics
    print()
    print(box_top())
    print(box_line("DESIGN METRICS"))
    print(box_mid())
    print(box_line(f"Cells:        {metrics['area_cells']:>12,}"))
    print(box_line(f"Area:         {metrics['area_um2']:>12.2f} µm²"))
    print(box_line(f"Wires:        {metrics['num_wires']:>12,}"))
    print(box_mid())
    print(box_line(f"Latency:      {metrics['latency_ps']:>12.0f} ps"))
    print(box_line(f"Target:       {metrics['target_clock_ps']:>12.0f} ps"))
    print(box_line(f"Slack:        {metrics['timing_slack_ps']:>+12.0f} ps"))
    print(box_mid())
    print(box_line(f"Energy/op:    {metrics['energy_pj']:>12.2f} pJ"))
    print(box_bot())

    # Cell breakdown
    cell_counts = metrics.get("cell_counts", {})
    if cell_counts:
        print("\n  Synthesized Cell Breakdown:")
        sorted_cells = sorted(cell_counts.items(), key=lambda x: -x[1])[:8]
        for cell_type, count in sorted_cells:
            pct = count / metrics['area_cells'] * 100
            print(f"    {cell_type:12s} {count:>6,}  ({pct:>5.1f}%)")

    # Technology Reference Table
    ref = technology["reference_areas"]
    print()
    print(box_top())
    print(box_line(f"TECHNOLOGY REFERENCE: {technology['name']}"))
    print(box_mid())
    print(box_line("Standard Cells"))
    print(box_line(f"  INV_X1:           {ref['INV_X1']:>10.3f} µm²"))
    print(box_line(f"  NAND2_X1:         {ref['NAND2_X1']:>10.3f} µm²"))
    print(box_line(f"  DFF (1-bit):      {ref['DFF_X1']:>10.3f} µm²"))
    print(box_mid())
    print(box_line("Registers"))
    print(box_line(f"  32-bit REG:       {ref['REG_32bit']:>10.2f} µm²"))
    print(box_mid())
    print(box_line("Arithmetic"))
    print(box_line(f"  32-bit Adder:     {ref['ADDER_32bit']:>10.1f} µm²   (ripple-carry)"))
    print(box_line(f"  32-bit Adder:     {ref['ADDER_32bit_CLA']:>10.1f} µm²   (carry-lookahead)"))
    print(box_line(f"  32-bit MUL:       {ref['MUL_32bit']:>10.1f} µm²   (array)"))
    print(box_line(f"  32-bit MUL:       {ref['MUL_32bit_BOOTH']:>10.1f} µm²   (Booth)"))
    print(box_mid())
    print(box_line("Aggregates"))
    print(box_line(f"  32-bit ALU:       {ref['ALU_32bit']:>10.1f} µm²"))
    print(box_line(f"  32-bit FPU:       {ref['FPU_32bit']:>10.1f} µm²"))
    print(box_mid())
    print(box_line("Memory"))
    print(box_line(f"  SRAM 1KB:         {ref['SRAM_1KB']:>10.1f} µm²"))
    print(box_line(f"  SRAM 32KB:        {ref['SRAM_32KB']:>10.1f} µm²"))
    print(box_bot())

    # Context: how does this design compare?
    design_area = metrics['area_um2']
    print("\n  Design Context:")
    print(f"    This design ({design_area:.1f} µm²) is equivalent to:")
    print(f"      • {design_area / ref['REG_32bit']:.1f}x 32-bit registers")
    print(f"      • {design_area / ref['ADDER_32bit']:.1f}x 32-bit adders")
    print(f"      • {design_area / ref['ALU_32bit']:.1f}x reference 32-bit ALUs")
    if design_area > ref['MUL_32bit']:
        print(f"      • {design_area / ref['MUL_32bit']:.2f}x 32-bit multipliers")

    print("\n" + "=" * 70)
    print("  HUMAN DECISION REQUIRED")
    print("=" * 70)
    print("""
  Review the metrics above in context of your system requirements.

  Options:
    • Accept: Design meets requirements → done
    • Optimize: Run architect loop → python workflow.py --optimize --with-llm
    • Adjust constraints and re-assess

  Consider:
    • Is this component the bottleneck, or are others worse?
    • Would 5-10% improvement here matter at system level?
    • What is the optimization effort vs. benefit?
""")
    print("=" * 70)


def run_langgraph_optimization(
    rtl_code: str,
    testbench: str,
    work_dir: Path,
    constraints: DesignConstraints,
    max_iterations: int,
    use_llm: bool,
    verbose: bool
):
    """Run optimization using LangGraph."""
    try:
        from graph import run_optimization
    except ImportError as e:
        print(f"Error importing graph module: {e}")
        print("Falling back to simple mode...")
        return run_simple_optimization(
            rtl_code, testbench, work_dir, constraints, max_iterations, verbose
        )

    # Create LLM client if requested
    llm_client = None
    if use_llm:
        try:
            from langchain_anthropic import ChatAnthropic
            llm_client = ChatAnthropic(model="claude-sonnet-4-20250514")
            print("Using Claude Sonnet for architect node")
        except ImportError:
            print("Warning: langchain-anthropic not installed, using mock mode")
        except Exception as e:
            print(f"Warning: Could not initialize LLM client: {e}")
            print("Using mock mode")

    # Run optimization
    try:
        result = run_optimization(
            rtl_code=rtl_code,
            top_module="riscv_alu",
            work_dir=work_dir,
            constraints=constraints.to_dict(),
            testbench=testbench,
            max_iterations=max_iterations,
            llm_client=llm_client,
            mock_mode=(llm_client is None),
            verbose=verbose,
        )
        return result
    except ImportError as e:
        print(f"LangGraph unavailable: {e}")
        print("Falling back to simple mode...")
        return run_simple_optimization(
            rtl_code, testbench, work_dir, constraints, max_iterations, verbose
        )


def run_simple_optimization(
    rtl_code: str,
    testbench: str,
    work_dir: Path,
    constraints: DesignConstraints,
    max_iterations: int,
    verbose: bool
):
    """Run optimization using simple loop (no LangGraph required)."""
    from graph import run_optimization_simple

    result = run_optimization_simple(
        rtl_code=rtl_code,
        top_module="riscv_alu",
        work_dir=work_dir,
        constraints=constraints.to_dict(),
        testbench=testbench,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    return result


def print_results(result: dict):
    """Print optimization results."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nStatus: {'SUCCESS' if result['success'] else 'INCOMPLETE'}")
    print(f"Iterations: {result['iterations']}")

    if result.get("baseline_metrics"):
        bm = result["baseline_metrics"]
        print(f"\nBaseline Metrics:")
        print(f"  Area: {bm.get('area_cells', 'N/A')} cells")
        print(f"  Wires: {bm.get('num_wires', 'N/A')}")

    if result.get("final_metrics"):
        fm = result["final_metrics"]
        print(f"\nFinal Metrics:")
        print(f"  Area: {fm.get('area_cells', 'N/A')} cells")
        print(f"  Wires: {fm.get('num_wires', 'N/A')}")

        # Calculate improvement
        if result.get("baseline_metrics"):
            baseline_area = result["baseline_metrics"].get("area_cells", 0)
            final_area = fm.get("area_cells", 0)
            if baseline_area > 0:
                improvement = (baseline_area - final_area) / baseline_area * 100
                print(f"\n  Area Change: {improvement:+.1f}%")

    # Print history summary
    if result.get("final_state") and result["final_state"].get("history"):
        history = result["final_state"]["history"]
        print(f"\nOptimization History ({len(history)} steps):")
        for step in history[-5:]:  # Last 5 steps
            agent = step.get("agent", "?")
            action = step.get("action", "?")
            success = "+" if step.get("success", False) else "-"
            reasoning = (step.get("reasoning") or step.get("error") or "")[:50]
            print(f"  [{success}] {agent}: {action} - {reasoning}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Human-in-the-Loop SoC Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Step 1: Baseline Assessment (default)
    python workflow.py                         # Get PPA metrics (7nm default)
    python workflow.py --technology 14nm       # Use 14nm process
    python workflow.py --technology 28nm       # Use 28nm process
    python workflow.py --target-clock 0.5      # Target 2GHz (500ps)

  Step 2: Optimization (if needed, after human review)
    python workflow.py --optimize --with-llm   # Claude-driven optimization
    python workflow.py --optimize --max-area 3000  # With area target

Workflow:
  1. Run assessment → review PPA metrics vs technology reference
  2. Compare to system-level requirements
  3. Decide: accept or optimize (--optimize flag)
        """
    )

    parser.add_argument(
        "--assess",
        action="store_true",
        help="Baseline assessment only - report PPA metrics for human review"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run optimization loop (requires --assess first or explicit target)"
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Use actual LLM for architect node (requires ANTHROPIC_API_KEY)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple loop instead of LangGraph"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum optimization iterations (default: 5)"
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=None,
        help="Maximum area constraint in cells (triggers optimization if exceeded)"
    )
    parser.add_argument(
        "--target-clock",
        type=float,
        default=1.0,
        help="Target clock period in ns (default: 1.0)"
    )
    parser.add_argument(
        "--technology",
        type=str,
        default=DEFAULT_TECHNOLOGY,
        choices=list(TECHNOLOGY_NODES.keys()),
        help=f"Process technology node (default: {DEFAULT_TECHNOLOGY})"
    )
    parser.add_argument(
        "--list-technologies",
        action="store_true",
        help="List all available technology nodes and exit"
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="./optimization_run",
        help="Working directory for outputs"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--no-testbench",
        action="store_true",
        help="Skip functional validation"
    )

    args = parser.parse_args()

    # Handle --list-technologies
    if args.list_technologies:
        print("\n" + "=" * 78)
        print("  AVAILABLE TECHNOLOGY NODES")
        print("=" * 78)
        print(f"\n  {'Node':<8} {'Name':<28} {'Generation':<18} {'Year':<6} {'PDK'}")
        print("  " + "-" * 74)

        # Sort by year descending
        sorted_techs = sorted(
            TECHNOLOGY_NODES.items(),
            key=lambda x: x[1].get("year", 0),
            reverse=True
        )
        for node, tech in sorted_techs:
            pdk = "Open" if tech.get("open_pdk") else ""
            print(f"  {node:<8} {tech['name']:<28} {tech.get('generation', ''):<18} {tech.get('year', ''):<6} {pdk}")

        print("\n  " + "-" * 74)
        print("  Open PDKs (full Yosys/OpenROAD support):")
        print("    • 130nm - SkyWater SKY130 (Google)")
        print("    • 180nm - GlobalFoundries GF180MCU")
        print("\n  Usage: python workflow.py --technology 28nm")
        print("=" * 78 + "\n")
        sys.exit(0)

    # Load designs
    rtl_code, testbench = load_designs()
    if args.no_testbench:
        testbench = None

    # Set up work directory
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    # =========================================================
    # HUMAN-IN-THE-LOOP FLOW
    # =========================================================
    # Default: baseline assessment (--assess or no flags)
    # Explicit: optimization loop (--optimize)
    # =========================================================

    if args.optimize:
        # OPTIMIZATION MODE: Run the architect loop
        print("=" * 60)
        print("  SoC OPTIMIZATION LOOP")
        print("=" * 60)
        print(f"  Mode: {'Simple loop' if args.simple else 'LangGraph'}")
        print(f"  LLM: {'Enabled' if args.with_llm else 'Mock mode'}")
        print(f"  Max iterations: {args.max_iter}")
        if args.max_area:
            print(f"  Target area: {args.max_area} cells")
        print("=" * 60 + "\n")

        print(f"Loaded RTL: {len(rtl_code)} characters")
        print(f"Loaded testbench: {len(testbench) if testbench else 0} characters")
        print(f"Work directory: {work_dir}\n")

        # Set up constraints
        constraints = DesignConstraints(
            target_clock_ns=args.target_clock,
            max_area_cells=args.max_area,
        )

        if args.simple:
            result = run_simple_optimization(
                rtl_code, testbench, work_dir, constraints, args.max_iter, verbose
            )
        else:
            result = run_langgraph_optimization(
                rtl_code, testbench, work_dir, constraints, args.max_iter,
                args.with_llm, verbose
            )

        # Print results
        print_results(result)
        sys.exit(0 if result["success"] else 1)

    else:
        # ASSESSMENT MODE (default): Baseline PPA for human review
        tech = TECHNOLOGY_NODES[args.technology]
        print("=" * 70)
        print("  SoC BASELINE ASSESSMENT")
        print("=" * 70)
        print(f"  Module:     riscv_alu")
        print(f"  Technology: {tech['name']}")
        print(f"  Target:     {args.target_clock} ns ({1000/args.target_clock:.0f} MHz)")
        print("=" * 70)

        result = run_baseline_assessment(
            rtl_code=rtl_code,
            testbench=testbench,
            top_module="riscv_alu",
            work_dir=work_dir,
            target_clock_ns=args.target_clock,
            technology=args.technology,
            verbose=verbose,
        )

        if result["success"]:
            print_ppa_report(result["metrics"], "riscv_alu", result["technology"])
            sys.exit(0)
        else:
            print("\n[ERROR] Assessment failed - fix issues before optimization")
            sys.exit(1)


if __name__ == "__main__":
    main()
