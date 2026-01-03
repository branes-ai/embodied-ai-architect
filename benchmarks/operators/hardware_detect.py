"""Auto-detect hardware platform and match against catalog.

This module wraps the comprehensive hardware detection from the `graphs` repository
when available, falling back to simplified detection otherwise.

The graphs auto_detect provides:
- SHA fingerprints for reproducible hw/sw identification
- Silicon-level details (CPU stepping, microcode, GPU VBIOS)
- Complete software stack versioning
- Environmental context (power mode, thermal state)
"""

import platform
import re
import subprocess
from pathlib import Path
from typing import Any

# Try to import graphs auto_detect for comprehensive detection
try:
    from graphs.hardware.calibration.auto_detect import (
        CalibrationContext,
        HardwareIdentity,
        SoftwareStack,
        EnvironmentalContext,
    )
    GRAPHS_AVAILABLE = True
except ImportError:
    GRAPHS_AVAILABLE = False
    CalibrationContext = None


def detect_hardware() -> dict[str, Any]:
    """Detect current hardware platform.

    Uses graphs.hardware.calibration.auto_detect when available for comprehensive
    detection with SHA fingerprints. Falls back to simplified detection otherwise.

    Returns:
        Dictionary with hardware info:
        - cpu_model: CPU model string
        - cpu_vendor: intel, amd, arm
        - gpu_model: GPU model if available
        - npu_available: Whether NPU is detected
        - hardware_id: Matched catalog ID or None
        - execution_targets: List of available targets (cpu, gpu, npu)

        When graphs is available, also includes:
        - hardware_fingerprint: SHA256 fingerprint of hardware identity
        - software_fingerprint: SHA256 fingerprint of software stack
        - calibration_context: Full CalibrationContext object
    """
    if GRAPHS_AVAILABLE:
        return _detect_with_graphs()
    else:
        return _detect_fallback()


def _detect_with_graphs() -> dict[str, Any]:
    """Comprehensive detection using graphs auto_detect."""
    context = CalibrationContext.detect()

    info = {
        # SHA fingerprints for reproducible identification
        "hardware_fingerprint": context.hardware.fingerprint,
        "software_fingerprint": context.software.fingerprint,
        "run_id": context.run_id,

        # CPU info from graphs
        "cpu_model": context.hardware.cpu.model,
        "cpu_vendor": _normalize_vendor(context.hardware.cpu.vendor),
        "cpu_stepping": context.hardware.cpu.stepping,
        "cpu_microcode": context.hardware.cpu.microcode,
        "cpu_cores_physical": context.hardware.cpu.cores_physical,
        "cpu_cores_logical": context.hardware.cpu.cores_logical,

        # GPU info from graphs
        "gpu_model": context.hardware.gpu.model if context.hardware.gpu else None,
        "gpu_vendor": context.hardware.gpu.vendor.lower() if context.hardware.gpu else None,
        "gpu_memory_mb": context.hardware.gpu.memory_mb if context.hardware.gpu else None,
        "cuda_available": context.software.cuda_version != "N/A",
        "rocm_available": context.software.rocm_version != "N/A",

        # Memory info
        "memory_gb": context.hardware.memory.total_gb,

        # Software stack versions
        "python_version": context.software.python_version,
        "pytorch_version": context.software.pytorch_version,
        "cuda_version": context.software.cuda_version,
        "cudnn_version": context.software.cudnn_version,

        # Environmental context
        "power_mode": context.environment.power_mode,
        "cpu_governor": context.environment.cpu_governor,
        "thermal_state": context.environment.thermal_state,

        # Full context for advanced use
        "calibration_context": context,

        # Execution targets
        "execution_targets": ["cpu"],
    }

    # Add GPU target if available
    if context.hardware.gpu:
        info["execution_targets"].append("gpu")

    # Detect NPU (not covered by graphs yet)
    npu_info = _detect_npu()
    info.update(npu_info)
    if npu_info.get("npu_available"):
        info["execution_targets"].append("npu")

    # Match against catalog
    info["hardware_id"] = _match_catalog(info)

    return info


def _detect_fallback() -> dict[str, Any]:
    """Simplified detection when graphs is not available."""
    info = {
        "cpu_model": None,
        "cpu_vendor": None,
        "gpu_model": None,
        "npu_available": False,
        "hardware_id": None,
        "execution_targets": ["cpu"],
        "hardware_fingerprint": None,  # Not available without graphs
        "software_fingerprint": None,
    }

    # Detect CPU
    info.update(_detect_cpu())

    # Detect GPU
    gpu_info = _detect_gpu()
    info.update(gpu_info)
    if gpu_info.get("gpu_model"):
        info["execution_targets"].append("gpu")

    # Detect NPU
    npu_info = _detect_npu()
    info.update(npu_info)
    if npu_info.get("npu_available"):
        info["execution_targets"].append("npu")

    # Match against catalog
    info["hardware_id"] = _match_catalog(info)

    return info


def _normalize_vendor(vendor: str) -> str:
    """Normalize vendor string to lowercase identifier."""
    vendor_lower = vendor.lower()
    if "intel" in vendor_lower or "genuineintel" in vendor_lower:
        return "intel"
    elif "amd" in vendor_lower or "authenticamd" in vendor_lower:
        return "amd"
    elif "arm" in vendor_lower:
        return "arm"
    elif "apple" in vendor_lower:
        return "apple"
    return vendor_lower


def _detect_cpu() -> dict[str, Any]:
    """Detect CPU model and vendor."""
    info = {"cpu_model": None, "cpu_vendor": None}

    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()

            # Extract model name
            match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            if match:
                info["cpu_model"] = match.group(1).strip()

            # Determine vendor
            model = info["cpu_model"] or ""
            if "AMD" in model:
                info["cpu_vendor"] = "amd"
            elif "Intel" in model:
                info["cpu_vendor"] = "intel"
            elif "ARM" in model or "aarch64" in platform.machine():
                info["cpu_vendor"] = "arm"

        except Exception:
            pass

    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info["cpu_model"] = result.stdout.strip()
                if "Apple" in info["cpu_model"]:
                    info["cpu_vendor"] = "apple"
                elif "Intel" in info["cpu_model"]:
                    info["cpu_vendor"] = "intel"
        except Exception:
            pass

    elif system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            info["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)

            model = info["cpu_model"] or ""
            if "AMD" in model:
                info["cpu_vendor"] = "amd"
            elif "Intel" in model:
                info["cpu_vendor"] = "intel"
        except Exception:
            pass

    return info


def _detect_gpu() -> dict[str, Any]:
    """Detect GPU availability and model."""
    info = {"gpu_model": None, "gpu_vendor": None, "cuda_available": False, "rocm_available": False}

    # Try PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_model"] = torch.cuda.get_device_name(0)
            if "NVIDIA" in info["gpu_model"]:
                info["gpu_vendor"] = "nvidia"
            elif "AMD" in info["gpu_model"] or "Radeon" in info["gpu_model"]:
                info["gpu_vendor"] = "amd"
                info["rocm_available"] = True
    except ImportError:
        pass

    # Try ROCm detection on Linux
    if not info["gpu_model"]:
        try:
            result = subprocess.run(
                ["rocminfo"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and "gfx" in result.stdout:
                info["rocm_available"] = True
                # Extract GPU name
                match = re.search(r"Marketing Name:\s*(.+)", result.stdout)
                if match:
                    info["gpu_model"] = match.group(1).strip()
                    info["gpu_vendor"] = "amd"
        except Exception:
            pass

    return info


def _detect_npu() -> dict[str, Any]:
    """Detect NPU availability."""
    info = {"npu_available": False, "npu_runtime": None}

    # Check ONNX Runtime providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()

        if "RyzenAIExecutionProvider" in providers:
            info["npu_available"] = True
            info["npu_runtime"] = "RyzenAI"
        elif "QNNExecutionProvider" in providers:
            info["npu_available"] = True
            info["npu_runtime"] = "QNN"
        elif "CoreMLExecutionProvider" in providers:
            # Apple Neural Engine
            info["npu_available"] = True
            info["npu_runtime"] = "CoreML"

    except ImportError:
        pass

    # Check for AMD XDNA driver on Linux
    if not info["npu_available"]:
        xdna_paths = [
            Path("/dev/accel/accel0"),  # XDNA device
            Path("/sys/class/accel"),
        ]
        for path in xdna_paths:
            if path.exists():
                info["npu_available"] = True
                info["npu_runtime"] = "XDNA"
                break

    return info


def _match_catalog(info: dict[str, Any]) -> str | None:
    """Match detected hardware against known catalog entries.

    Returns:
        Hardware ID from catalog or None
    """
    cpu_model = info.get("cpu_model", "") or ""

    # AMD Ryzen AI NUCs
    if "8945HS" in cpu_model:
        return "amd_ryzen_9_8945hs_nuc"
    if "8845HS" in cpu_model:
        return "amd_ryzen_7_8845hs_nuc"
    if "7940HS" in cpu_model:
        return "amd_ryzen_9_7940hs"
    if "7840HS" in cpu_model:
        return "amd_ryzen_7_7840hs"

    # NVIDIA Jetson
    if "Orin" in cpu_model or _is_jetson():
        if _is_jetson_agx():
            return "Jetson-Orin-AGX"
        return "Jetson-Orin-Nano"

    # Intel
    if "13900" in cpu_model:
        return "intel_core_i9_13900k"
    if "12900" in cpu_model:
        return "intel_core_i9_12900k"

    # Generic fallback based on vendor
    vendor = info.get("cpu_vendor")
    if vendor == "amd":
        return "generic_amd_cpu"
    if vendor == "intel":
        return "generic_intel_cpu"
    if vendor == "arm":
        return "generic_arm_cpu"

    return None


def _is_jetson() -> bool:
    """Check if running on NVIDIA Jetson."""
    try:
        with open("/etc/nv_tegra_release") as f:
            return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["cat", "/proc/device-tree/model"],
            capture_output=True,
            text=True,
        )
        return "NVIDIA" in result.stdout and "Jetson" in result.stdout
    except Exception:
        pass

    return False


def _is_jetson_agx() -> bool:
    """Check if running on Jetson AGX variant."""
    try:
        result = subprocess.run(
            ["cat", "/proc/device-tree/model"],
            capture_output=True,
            text=True,
        )
        return "AGX" in result.stdout
    except Exception:
        pass
    return False


def print_hardware_info(info: dict[str, Any]) -> None:
    """Print detected hardware info."""
    print("Detected Hardware:")

    # Show SHA fingerprints if available (from graphs)
    if info.get("hardware_fingerprint"):
        print(f"  HW Fingerprint: {info['hardware_fingerprint']}")
    if info.get("software_fingerprint"):
        print(f"  SW Fingerprint: {info['software_fingerprint']}")

    print(f"  CPU: {info.get('cpu_model', 'Unknown')}")
    print(f"  Vendor: {info.get('cpu_vendor', 'Unknown')}")

    # Show detailed CPU info if available (from graphs)
    if info.get("cpu_stepping") is not None:
        print(f"  Stepping: {info['cpu_stepping']}, Microcode: {info.get('cpu_microcode', 'N/A')}")
    if info.get("cpu_cores_physical"):
        print(f"  Cores: {info['cpu_cores_physical']}P / {info.get('cpu_cores_logical', '?')}L")

    if info.get("gpu_model"):
        print(f"  GPU: {info['gpu_model']}")
        if info.get("gpu_memory_mb"):
            print(f"  GPU Memory: {info['gpu_memory_mb']} MB")
        if info.get("cuda_available"):
            cuda_ver = info.get("cuda_version", "")
            print(f"  CUDA: Available ({cuda_ver})" if cuda_ver and cuda_ver != "N/A" else "  CUDA: Available")
        if info.get("rocm_available"):
            print("  ROCm: Available")

    if info.get("npu_available"):
        print(f"  NPU: Available ({info.get('npu_runtime', 'Unknown runtime')})")

    if info.get("memory_gb"):
        print(f"  Memory: {info['memory_gb']:.1f} GB")

    # Environmental context if available (from graphs)
    if info.get("power_mode") and info["power_mode"] != "TDP":
        print(f"  Power Mode: {info['power_mode']}")
    if info.get("thermal_state") and info["thermal_state"] not in ("unknown", "cool"):
        print(f"  Thermal: {info['thermal_state']}")

    print(f"  Execution targets: {', '.join(info.get('execution_targets', ['cpu']))}")

    if info.get("hardware_id"):
        print(f"  Catalog match: {info['hardware_id']}")
    else:
        print("  Catalog match: No match found (using detected specs)")

    # Note if using graphs or fallback
    if not info.get("hardware_fingerprint"):
        print("  Note: Install 'graphs' package for full hw/sw fingerprinting")


def get_calibration_context():
    """Get the full CalibrationContext from graphs if available.

    Returns:
        CalibrationContext object or None if graphs is not installed.
        The context includes:
        - hardware.fingerprint: SHA256 of hw identity
        - software.fingerprint: SHA256 of sw stack
        - environment: Power mode, thermal state, etc.
    """
    if not GRAPHS_AVAILABLE:
        return None
    return CalibrationContext.detect()


if __name__ == "__main__":
    info = detect_hardware()
    print_hardware_info(info)
