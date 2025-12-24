"""
RTL Synthesis Tool - Generate gate-level netlist and extract PPA metrics.

Uses Yosys with a generic cell library for technology-independent synthesis.
Extracts area (cell count) and structural metrics.
"""

import subprocess
import re
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict


@dataclass
class SynthesisResult:
    """Results from RTL synthesis."""
    success: bool
    area_cells: int = 0
    area_um2: float = 0.0
    num_wires: int = 0
    num_cells: int = 0
    cell_counts: Dict[str, int] = field(default_factory=dict)
    netlist_path: Optional[str] = None
    log_excerpt: str = ""
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class RTLSynthesisTool:
    """
    Synthesize RTL to gate-level netlist using Yosys.

    Performs technology-independent synthesis and extracts:
    - Cell count and breakdown
    - Wire count
    - Estimated area

    For more accurate PPA, use with a technology library (.lib file).
    """

    def __init__(self, work_dir: Optional[Path] = None):
        self.work_dir = (work_dir or Path(tempfile.mkdtemp())).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        rtl_source: str,
        top_module: str,
        optimize: bool = True,
        liberty_file: Optional[str] = None
    ) -> dict:
        """Synthesize RTL and extract metrics."""
        # Write RTL
        rtl_path = self.work_dir / f"{top_module}.sv"
        rtl_path.write_text(rtl_source)

        # Generate synthesis script
        script = self._generate_script(rtl_path, top_module, optimize, liberty_file)
        script_path = self.work_dir / "synth.ys"
        script_path.write_text(script)

        try:
            proc = subprocess.run(
                ["yosys", "-s", str(script_path)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.work_dir
            )

            if proc.returncode != 0:
                errors = [l for l in proc.stderr.splitlines() if "error" in l.lower()]
                if not errors:
                    errors = [proc.stderr[:500] if proc.stderr else "Unknown synthesis error"]
                return SynthesisResult(
                    success=False,
                    errors=errors,
                    log_excerpt=proc.stdout[-1000:] if proc.stdout else ""
                ).to_dict()

            # Parse stats from output
            result = self._parse_stats(proc.stdout)
            result.netlist_path = str(self.work_dir / "netlist.json")
            result.log_excerpt = proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout

            return result.to_dict()

        except FileNotFoundError:
            return self._mock_synthesis(rtl_source, top_module).to_dict()
        except subprocess.TimeoutExpired:
            return SynthesisResult(
                success=False,
                errors=["Synthesis timed out after 120s"]
            ).to_dict()

    def _generate_script(
        self,
        rtl_path: Path,
        top_module: str,
        optimize: bool,
        liberty_file: Optional[str]
    ) -> str:
        """Generate Yosys synthesis script."""
        opt_cmds = "opt; opt_clean; opt -full;" if optimize else "opt;"

        # Base synthesis
        script = f"""
# Read RTL
read_verilog -sv {rtl_path}
hierarchy -check -top {top_module}

# Generic synthesis
proc; {opt_cmds}
fsm; {opt_cmds}
memory; {opt_cmds}
techmap; {opt_cmds}
"""
        # Technology mapping
        if liberty_file and Path(liberty_file).exists():
            script += f"""
# Technology-specific synthesis
abc -liberty {liberty_file}
"""
        else:
            script += f"""
# Generic gate mapping (no technology library)
abc -g AND,OR,XOR,NAND,NOR,XNOR,MUX,AOI3,OAI3,AOI4,OAI4
{opt_cmds}
"""
        # Statistics and output
        script += f"""
# Statistics
stat

# Write outputs
write_json {self.work_dir / 'netlist.json'}
write_verilog {self.work_dir / 'synth.v'}
"""
        return script

    def _parse_stats(self, stdout: str) -> SynthesisResult:
        """Parse Yosys stat output."""
        result = SynthesisResult(success=True)
        cell_counts = {}

        in_stats = False
        for line in stdout.splitlines():
            if "Printing statistics" in line or line.strip().startswith("=== "):
                in_stats = True
                continue

            if not in_stats:
                continue

            line = line.strip()

            # Number of wires - matches "Number of wires:" or just "123 wires"
            if "Number of wires:" in line:
                match = re.search(r'(\d+)', line)
                if match:
                    result.num_wires = int(match.group(1))
            elif line.endswith("wires") and not "wire bits" in line and not "public" in line:
                match = re.search(r'(\d+)\s+wires', line)
                if match:
                    result.num_wires = int(match.group(1))

            # Number of cells - matches "Number of cells:" or just "123 cells"
            elif "Number of cells:" in line:
                match = re.search(r'(\d+)', line)
                if match:
                    result.num_cells = int(match.group(1))
            elif line.endswith("cells"):
                match = re.search(r'(\d+)\s+cells', line)
                if match:
                    result.num_cells = int(match.group(1))

            # Individual cell types - matches "123   $_AND_" format (number before cell name)
            elif "$_" in line:
                cell_match = re.search(r'(\d+)\s+\$_(\w+)_', line)
                if cell_match:
                    cell_counts[cell_match.group(2)] = int(cell_match.group(1))

        result.cell_counts = cell_counts
        result.area_cells = sum(cell_counts.values()) if cell_counts else result.num_cells
        # Rough area estimate: 0.5 um^2 per cell (very rough)
        result.area_um2 = result.area_cells * 0.5

        return result

    def _mock_synthesis(self, rtl: str, top: str) -> SynthesisResult:
        """Mock synthesis when Yosys not available."""
        # Estimate complexity from RTL
        num_always = rtl.count("always")
        num_assign = rtl.count("assign")
        num_ops = len(re.findall(r'[\+\-\*\/\&\|\^]', rtl))

        estimated_cells = (num_always * 10) + (num_assign * 3) + (num_ops * 2)
        estimated_cells = max(estimated_cells, 10)  # Minimum

        return SynthesisResult(
            success=True,
            area_cells=estimated_cells,
            area_um2=estimated_cells * 0.5,
            num_wires=estimated_cells * 2,
            num_cells=estimated_cells,
            cell_counts={"estimated": estimated_cells},
            log_excerpt="Mock synthesis - Yosys not available",
            warnings=["Using mock synthesis - install Yosys for real metrics"]
        )
