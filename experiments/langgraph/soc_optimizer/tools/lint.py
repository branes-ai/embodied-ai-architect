"""
RTL Linting Tool - Fast syntax checking before synthesis.

Uses Verilator in lint-only mode for sub-second feedback.
This catches syntax errors before wasting time on expensive synthesis.
"""

import subprocess
import re
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class LintResult:
    """Results from RTL linting."""
    success: bool
    module_name: str = ""
    errors: List[str] = None
    warnings: List[str] = None
    ports: List[dict] = None
    parameters: List[dict] = None

    def __post_init__(self):
        self.errors = self.errors or []
        self.warnings = self.warnings or []
        self.ports = self.ports or []
        self.parameters = self.parameters or []

    def to_dict(self) -> dict:
        return asdict(self)


class RTLLintTool:
    """
    Lint and parse RTL to extract structure.

    Uses Verilator for fast linting, falls back to Yosys if unavailable.
    If neither is available, performs basic regex-based parsing.
    """

    def __init__(self, work_dir: Optional[Path] = None):
        self.work_dir = (work_dir or Path(tempfile.mkdtemp())).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(self, rtl_source: str) -> dict:
        """Lint RTL and extract structure."""
        # Write RTL to temp file
        rtl_path = self.work_dir / "design.sv"
        rtl_path.write_text(rtl_source)

        # Try Verilator first (fastest)
        result = self._try_verilator(rtl_path, rtl_source)
        if result is not None:
            return result.to_dict()

        # Fall back to Yosys
        result = self._try_yosys(rtl_path, rtl_source)
        if result is not None:
            return result.to_dict()

        # Fall back to regex parsing
        return self._regex_parse(rtl_source).to_dict()

    def _try_verilator(self, rtl_path: Path, rtl_source: str) -> Optional[LintResult]:
        """Try linting with Verilator."""
        try:
            proc = subprocess.run(
                ["verilator", "--lint-only", "-Wall", "-Wno-fatal", str(rtl_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.work_dir
            )

            errors = []
            warnings = []

            # Parse Verilator output
            for line in proc.stderr.splitlines():
                if "%Error" in line:
                    errors.append(line.strip())
                elif "%Warning" in line:
                    warnings.append(line.strip())

            # Extract module info with regex
            module_name, ports, params = self._extract_module_info(rtl_source)

            return LintResult(
                success=(proc.returncode == 0 and len(errors) == 0),
                module_name=module_name,
                errors=errors,
                warnings=warnings,
                ports=ports,
                parameters=params
            )

        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return LintResult(
                success=False,
                errors=["Verilator lint timed out after 30s"]
            )

    def _try_yosys(self, rtl_path: Path, rtl_source: str) -> Optional[LintResult]:
        """Try linting with Yosys."""
        script = f"""
read_verilog -sv {rtl_path}
hierarchy -check
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

            errors = []
            warnings = []

            for line in proc.stderr.splitlines() + proc.stdout.splitlines():
                if "ERROR:" in line or "error:" in line.lower():
                    errors.append(line.strip())
                elif "Warning:" in line:
                    warnings.append(line.strip())

            module_name, ports, params = self._extract_module_info(rtl_source)

            return LintResult(
                success=(proc.returncode == 0 and len(errors) == 0),
                module_name=module_name,
                errors=errors,
                warnings=warnings,
                ports=ports,
                parameters=params
            )

        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return LintResult(
                success=False,
                errors=["Yosys lint timed out after 30s"]
            )

    def _regex_parse(self, rtl_source: str) -> LintResult:
        """Fall back to regex-based parsing when no tools available."""
        errors = []

        module_name, ports, params = self._extract_module_info(rtl_source)

        # Basic structural checks
        if not module_name:
            errors.append("No module definition found")
        if rtl_source.count("module") != rtl_source.count("endmodule"):
            errors.append("Mismatched module/endmodule")
        if rtl_source.count("begin") != rtl_source.count("end"):
            errors.append("Mismatched begin/end blocks")

        return LintResult(
            success=(len(errors) == 0),
            module_name=module_name,
            errors=errors,
            warnings=["Using regex-based parsing - EDA tools not available"],
            ports=ports,
            parameters=params
        )

    def _extract_module_info(self, rtl: str) -> tuple:
        """Extract module name, ports, and parameters from RTL."""
        # Module name
        module_match = re.search(r'module\s+(\w+)', rtl)
        module_name = module_match.group(1) if module_match else ""

        # Ports
        port_pattern = r'(input|output|inout)\s+(?:wire|reg|logic)?\s*(?:\[[\d:]+\])?\s*(\w+)'
        ports = [{"direction": p[0], "name": p[1]} for p in re.findall(port_pattern, rtl)]

        # Parameters
        param_pattern = r'parameter\s+(\w+)\s*=\s*([^,;]+)'
        params = [{"name": p[0], "value": p[1].strip()} for p in re.findall(param_pattern, rtl)]

        return module_name, ports, params
