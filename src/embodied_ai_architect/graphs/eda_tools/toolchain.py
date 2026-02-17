"""Unified EDA toolchain facade.

Provides a single entry point for lint, synthesis, and simulation
with automatic tool detection and fallback.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from embodied_ai_architect.graphs.eda_tools.lint import RTLLintTool
from embodied_ai_architect.graphs.eda_tools.synthesis import AUTO_TIMEOUT, RTLSynthesisTool
from embodied_ai_architect.graphs.eda_tools.simulation import SimulationTool


class EDAToolchain:
    """Unified facade for EDA tools.

    Manages work directories and provides access to lint, synthesis,
    and simulation tools with automatic tool availability detection.

    Usage:
        toolchain = EDAToolchain(process_nm=28)
        lint_result = toolchain.lint(rtl_source)
        synth_result = toolchain.synthesize(rtl_source, "my_module")
        sim_result = toolchain.simulate(rtl_source, tb_source, "my_module")

    Args:
        work_dir: Root working directory for intermediate files.
        process_nm: Technology node for area scaling.
        synth_timeout: Yosys timeout in seconds.
            AUTO_TIMEOUT (-1, default) estimates from RTL complexity.
    """

    def __init__(
        self,
        work_dir: Optional[Path] = None,
        process_nm: int = 28,
        synth_timeout: int = AUTO_TIMEOUT,
    ):
        self.work_dir = (work_dir or Path(tempfile.mkdtemp(prefix="eda_"))).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.process_nm = process_nm

        self._lint_tool = RTLLintTool(self.work_dir / "lint")
        self._synth_tool = RTLSynthesisTool(
            self.work_dir / "synth", process_nm=process_nm, timeout=synth_timeout
        )
        self._sim_tool = SimulationTool(self.work_dir / "sim")

    @property
    def available_tools(self) -> dict[str, bool]:
        """Check which EDA tools are available on the system."""
        return {
            "verilator": shutil.which("verilator") is not None,
            "yosys": shutil.which("yosys") is not None,
            "iverilog": shutil.which("iverilog") is not None,
        }

    def lint(self, rtl_source: str) -> dict:
        """Lint RTL source code."""
        return self._lint_tool.run(rtl_source)

    def synthesize(
        self,
        rtl_source: str,
        top_module: str,
        optimize: bool = True,
        liberty_file: Optional[str] = None,
    ) -> dict:
        """Synthesize RTL to gate-level netlist."""
        return self._synth_tool.run(rtl_source, top_module, optimize, liberty_file)

    def simulate(
        self,
        rtl_source: str,
        testbench_source: str,
        top_module: str,
        generate_vcd: bool = False,
    ) -> dict:
        """Run functional simulation."""
        return self._sim_tool.run(rtl_source, testbench_source, top_module, generate_vcd)
