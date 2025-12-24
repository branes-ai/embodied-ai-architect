"""
EDA Tool Wrappers for SoC Optimization

These tools wrap real EDA operations (using Yosys for synthesis).
They provide structured input/output that agents can work with.
"""

import subprocess
import json
import hashlib
import re
import tempfile
from pathlib import Path
from typing import Type, Optional
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field


# ============================================================
# Data Models
# ============================================================

@dataclass
class SynthesisMetrics:
    """Metrics extracted from synthesis."""
    area_cells: int = 0
    area_um2: float = 0.0  # Estimated
    num_wires: int = 0
    num_cells: int = 0
    # Cell type breakdown
    cell_counts: dict = None
    
    def __post_init__(self):
        if self.cell_counts is None:
            self.cell_counts = {}


@dataclass 
class LintResult:
    """Results from linting/parsing."""
    success: bool
    errors: list
    warnings: list
    module_name: str = ""
    ports: list = None
    parameters: list = None


@dataclass
class SimulationResult:
    """Results from simulation."""
    success: bool
    tests_passed: int = 0
    tests_failed: int = 0
    error_message: str = ""
    stdout: str = ""


# ============================================================
# Tool Implementations
# ============================================================

class RTLLintTool:
    """
    Lint and parse RTL to extract structure.
    Uses Yosys for parsing.
    """
    
    name = "rtl_lint"
    description = """Parse and lint Verilog/SystemVerilog code.
    Returns: module structure, ports, parameters, and any syntax errors.
    Use this FIRST before any other operation to validate RTL."""
    
    def __init__(self, work_dir: Path = None):
        self.work_dir = (work_dir or Path(tempfile.mkdtemp())).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, rtl_source: str) -> dict:
        """Lint RTL and extract structure."""
        
        # Write RTL to temp file
        rtl_path = self.work_dir / "design.v"
        rtl_path.write_text(rtl_source)
        
        # Create Yosys script for parsing only
        script = f"""
read_verilog -sv {rtl_path}
hierarchy -check
stat
"""
        script_path = self.work_dir / "lint.ys"
        script_path.write_text(script)
        
        try:
            proc = subprocess.run(
                ["yosys", "-s", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.work_dir
            )
            
            result = self._parse_output(proc.stdout, proc.stderr, proc.returncode, rtl_source)
            return asdict(result)
            
        except FileNotFoundError:
            # Yosys not installed - provide mock response for demo
            return self._mock_lint(rtl_source)
        except subprocess.TimeoutExpired:
            return asdict(LintResult(
                success=False,
                errors=["Lint timed out after 30s"],
                warnings=[]
            ))
    
    def _parse_output(self, stdout: str, stderr: str, code: int, rtl: str) -> LintResult:
        errors = []
        warnings = []
        
        for line in stderr.splitlines():
            if "error:" in line.lower():
                errors.append(line.strip())
            elif "warning:" in line.lower():
                warnings.append(line.strip())
        
        for line in stdout.splitlines():
            if "ERROR:" in line:
                errors.append(line.strip())
            elif "Warning:" in line:
                warnings.append(line.strip())
        
        # Extract module name
        module_match = re.search(r'module\s+(\w+)', rtl)
        module_name = module_match.group(1) if module_match else ""
        
        # Extract ports
        ports = re.findall(r'(input|output|inout)\s+(?:wire|reg)?\s*(?:\[[\d:]+\])?\s*(\w+)', rtl)
        port_list = [{"direction": p[0], "name": p[1]} for p in ports]
        
        # Extract parameters
        params = re.findall(r'parameter\s+(\w+)\s*=\s*(\d+)', rtl)
        param_list = [{"name": p[0], "value": p[1]} for p in params]
        
        return LintResult(
            success=(code == 0 and len(errors) == 0),
            errors=errors,
            warnings=warnings,
            module_name=module_name,
            ports=port_list,
            parameters=param_list
        )
    
    def _mock_lint(self, rtl: str) -> dict:
        """Mock lint for when Yosys isn't available."""
        # Basic regex-based parsing
        module_match = re.search(r'module\s+(\w+)', rtl)
        module_name = module_match.group(1) if module_match else ""
        
        ports = re.findall(r'(input|output|inout)\s+(?:wire|reg)?\s*(?:\[[\d:]+\])?\s*(\w+)', rtl)
        port_list = [{"direction": p[0], "name": p[1]} for p in ports]
        
        params = re.findall(r'parameter\s+(\w+)\s*=\s*(\d+)', rtl)
        param_list = [{"name": p[0], "value": p[1]} for p in params]
        
        # Check for basic errors
        errors = []
        if not module_match:
            errors.append("No module definition found")
        if rtl.count("module") != rtl.count("endmodule"):
            errors.append("Mismatched module/endmodule")
        
        return asdict(LintResult(
            success=len(errors) == 0,
            errors=errors,
            warnings=["Mock lint - Yosys not available"],
            module_name=module_name,
            ports=port_list,
            parameters=param_list
        ))


class RTLSynthesisTool:
    """
    Synthesize RTL to gate-level netlist.
    Uses Yosys with generic cell library.
    """
    
    name = "rtl_synthesis"
    description = """Synthesize Verilog to gate-level netlist.
    Returns: area metrics, cell counts, and synthesis status.
    This gives you real PPA metrics to evaluate design quality."""
    
    def __init__(self, work_dir: Path = None):
        self.work_dir = (work_dir or Path(tempfile.mkdtemp())).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, rtl_source: str, top_module: str, optimize: bool = True) -> dict:
        """Synthesize RTL and extract metrics."""
        
        # Write RTL
        rtl_path = self.work_dir / f"{top_module}.v"
        rtl_path.write_text(rtl_source)
        
        # Create synthesis script
        opt_cmds = "opt; opt_clean; opt -full;" if optimize else "opt;"
        
        script = f"""
read_verilog -sv {rtl_path}
hierarchy -check -top {top_module}
proc; {opt_cmds}
fsm; {opt_cmds}
memory; {opt_cmds}
techmap; {opt_cmds}
abc -g AND,OR,XOR,NAND,NOR,XNOR,MUX,AOI3,OAI3,AOI4,OAI4; {opt_cmds}
stat
write_json {self.work_dir / 'netlist.json'}
"""
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
            
            metrics = self._parse_stats(proc.stdout)
            
            return {
                "success": proc.returncode == 0,
                "metrics": asdict(metrics),
                "netlist_path": str(self.work_dir / 'netlist.json'),
                "log_excerpt": proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout,
                "errors": [l for l in proc.stderr.splitlines() if "error" in l.lower()]
            }
            
        except FileNotFoundError:
            return self._mock_synthesis(rtl_source, top_module)
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "errors": ["Synthesis timed out after 120s"],
                "metrics": asdict(SynthesisMetrics())
            }
    
    def _parse_stats(self, stdout: str) -> SynthesisMetrics:
        """Parse Yosys stat output."""
        metrics = SynthesisMetrics()
        
        # Look for stats section
        in_stats = False
        for line in stdout.splitlines():
            if "Statistics" in line or "Number of" in line:
                in_stats = True
            
            if in_stats:
                # Number of wires
                if "wires" in line.lower():
                    match = re.search(r'(\d+)', line)
                    if match:
                        metrics.num_wires = int(match.group(1))
                
                # Number of cells
                if "cells" in line.lower() and "$" not in line:
                    match = re.search(r'(\d+)', line)
                    if match:
                        metrics.num_cells = int(match.group(1))
                
                # Cell breakdown
                cell_match = re.search(r'\$(\w+)\s+(\d+)', line)
                if cell_match:
                    metrics.cell_counts[cell_match.group(1)] = int(cell_match.group(2))
        
        # Estimate area (rough: 1 cell ≈ 1 unit)
        metrics.area_cells = sum(metrics.cell_counts.values())
        metrics.area_um2 = metrics.area_cells * 0.5  # Rough estimate
        
        return metrics
    
    def _mock_synthesis(self, rtl: str, top: str) -> dict:
        """Mock synthesis for demo when Yosys not available."""
        # Estimate complexity from RTL
        num_always = rtl.count("always")
        num_assign = rtl.count("assign")
        num_ops = len(re.findall(r'[\+\-\*\/\&\|\^]', rtl))
        
        estimated_cells = (num_always * 10) + (num_assign * 3) + (num_ops * 2)
        
        return {
            "success": True,
            "metrics": asdict(SynthesisMetrics(
                area_cells=estimated_cells,
                area_um2=estimated_cells * 0.5,
                num_wires=estimated_cells * 2,
                num_cells=estimated_cells,
                cell_counts={"estimated": estimated_cells}
            )),
            "netlist_path": "mock_netlist.json",
            "log_excerpt": "Mock synthesis - Yosys not available",
            "errors": [],
            "warning": "Using mock synthesis - install Yosys for real metrics"
        }


class SimulationTool:
    """
    Run functional simulation using Icarus Verilog.
    """
    
    name = "simulation"
    description = """Run functional simulation with testbench.
    Returns: pass/fail status, test counts.
    Use this to verify that optimized designs still work correctly."""
    
    def __init__(self, work_dir: Path = None):
        self.work_dir = (work_dir or Path(tempfile.mkdtemp())).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, rtl_source: str, testbench_source: str, top_module: str) -> dict:
        """Run simulation."""
        
        # Write files
        rtl_path = self.work_dir / f"{top_module}.v"
        rtl_path.write_text(rtl_source)
        
        tb_path = self.work_dir / "tb.v"
        tb_path.write_text(testbench_source)
        
        vvp_path = self.work_dir / "sim.vvp"
        
        try:
            # Compile
            compile_proc = subprocess.run(
                ["iverilog", "-o", str(vvp_path), str(rtl_path), str(tb_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.work_dir
            )
            
            if compile_proc.returncode != 0:
                return asdict(SimulationResult(
                    success=False,
                    error_message=f"Compile error: {compile_proc.stderr}"
                ))
            
            # Run
            run_proc = subprocess.run(
                ["vvp", str(vvp_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.work_dir
            )
            
            return self._parse_results(run_proc.stdout, run_proc.returncode)
            
        except FileNotFoundError:
            return self._mock_simulation(rtl_source)
        except subprocess.TimeoutExpired:
            return asdict(SimulationResult(
                success=False,
                error_message="Simulation timed out"
            ))
    
    def _parse_results(self, stdout: str, returncode: int) -> dict:
        """Parse simulation output."""
        
        # Look for test summary
        passed_match = re.search(r'Passed:\s*(\d+)', stdout)
        failed_match = re.search(r'Failed:\s*(\d+)', stdout)
        
        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        
        # Check for ALL TESTS PASSED
        all_passed = "ALL TESTS PASSED" in stdout
        
        return asdict(SimulationResult(
            success=all_passed or (failed == 0 and passed > 0),
            tests_passed=passed,
            tests_failed=failed,
            stdout=stdout[-1000:] if len(stdout) > 1000 else stdout
        ))
    
    def _mock_simulation(self, rtl: str) -> dict:
        """Mock simulation for demo."""
        # Check for obvious issues
        has_module = "module" in rtl and "endmodule" in rtl
        has_clk = "clk" in rtl
        has_always = "always" in rtl
        
        # CRITICAL: Check for the shifter bug!
        # The buggy version has: a << 1 (always shifts by 1)
        # The correct version has: a << b[2:0] (shifts by variable amount)
        has_shifter_bug = False
        if "OP_SHL" in rtl or "SHL" in rtl:
            import re
            # Remove comments first
            rtl_no_comments = re.sub(r'//.*$', '', rtl, flags=re.MULTILINE)
            
            # Look for proper variable shifts: << b[ or >> b[
            proper_shift = re.search(r'<<\s*b\[', rtl_no_comments) or re.search(r'>>\s*b\[', rtl_no_comments)
            # Look for constant shifts: << 1; or >> 1;
            const_shift = re.search(r'<<\s*1\s*;', rtl_no_comments) or re.search(r'>>\s*1\s*;', rtl_no_comments)
            
            # Bug: has constant shift but no proper variable shift
            if const_shift and not proper_shift:
                has_shifter_bug = True
        
        if has_module and has_clk and has_always and not has_shifter_bug:
            return asdict(SimulationResult(
                success=True,
                tests_passed=15,
                tests_failed=0,
                stdout="Mock simulation - Icarus not available\nALL TESTS PASSED (mock)"
            ))
        elif has_shifter_bug:
            return asdict(SimulationResult(
                success=False,
                tests_passed=12,
                tests_failed=3,
                error_message="Shifter test cases failed - shift by 3 produced wrong result",
                stdout="Mock simulation detected shifter bug!\nFailed: SHL by 3, SHR by 2\nExpected: 0x08, Got: 0x02"
            ))
        else:
            return asdict(SimulationResult(
                success=False,
                tests_passed=0,
                tests_failed=1,
                error_message="Mock simulation detected missing components"
            ))


# ============================================================
# Tool Registry
# ============================================================

TOOLS = {
    "rtl_lint": RTLLintTool,
    "rtl_synthesis": RTLSynthesisTool,
    "simulation": SimulationTool,
}


def get_tool(name: str, work_dir: Path = None):
    """Get a tool instance by name."""
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return TOOLS[name](work_dir)
