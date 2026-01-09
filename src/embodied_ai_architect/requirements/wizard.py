"""Interactive requirements wizard.

Guides users through defining pipeline requirements via an
interactive CLI interface.
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from .models import (
    DeploymentRequirements,
    ExecutionTarget,
    HardwareRequirements,
    PerceptionRequirements,
    PipelineRequirements,
    TaskType,
)

console = Console()


class RequirementsWizard:
    """Interactive wizard for creating pipeline requirements.

    Guides users through a step-by-step process to define
    perception tasks, accuracy constraints, and hardware requirements.
    """

    def __init__(self):
        self.requirements: Optional[PipelineRequirements] = None

    def run(self) -> PipelineRequirements:
        """Run the interactive wizard.

        Returns:
            Configured PipelineRequirements
        """
        console.print()
        console.print(
            Panel.fit(
                "[bold blue]Pipeline Requirements Wizard[/bold blue]\n"
                "Define your perception pipeline requirements step by step.",
                border_style="blue",
            )
        )
        console.print()

        # Step 1: Basic info
        name = self._get_name()
        description = self._get_description()

        # Step 2: Perception tasks
        perception = self._get_perception_requirements()

        # Step 3: Hardware constraints
        hardware = self._get_hardware_requirements()

        # Step 4: Deployment config
        deployment = self._get_deployment_requirements()

        self.requirements = PipelineRequirements(
            name=name,
            description=description,
            perception=perception,
            hardware=hardware,
            deployment=deployment,
        )

        # Show summary
        self._show_summary()

        return self.requirements

    def _get_name(self) -> str:
        """Get pipeline name."""
        console.print("[bold]Step 1: Basic Information[/bold]")
        name = Prompt.ask(
            "  Pipeline name",
            default="my-pipeline",
        )
        return name.replace(" ", "-").lower()

    def _get_description(self) -> Optional[str]:
        """Get optional description."""
        description = Prompt.ask(
            "  Description (optional)",
            default="",
        )
        return description if description else None

    def _get_perception_requirements(self) -> PerceptionRequirements:
        """Get perception task requirements."""
        console.print()
        console.print("[bold]Step 2: Perception Tasks[/bold]")

        # Show available tasks
        self._show_task_options()

        # Get tasks
        tasks = self._select_tasks()

        # Get target classes if detection task
        target_classes = []
        if TaskType.OBJECT_DETECTION in tasks:
            classes_input = Prompt.ask(
                "  Target classes (comma-separated, e.g., person,car,dog)",
                default="",
            )
            if classes_input:
                target_classes = [c.strip() for c in classes_input.split(",")]

        # Get accuracy constraint
        min_accuracy = None
        if Confirm.ask("  Set minimum accuracy constraint?", default=False):
            accuracy_pct = FloatPrompt.ask(
                "    Minimum accuracy (%)",
                default=80.0,
            )
            min_accuracy = accuracy_pct / 100.0

        # Get latency constraint
        max_latency_ms = None
        if Confirm.ask("  Set maximum latency constraint?", default=False):
            max_latency_ms = FloatPrompt.ask(
                "    Maximum latency (ms)",
                default=100.0,
            )

        # Get FPS constraint
        min_fps = None
        if Confirm.ask("  Set minimum FPS constraint?", default=False):
            min_fps = FloatPrompt.ask(
                "    Minimum FPS",
                default=30.0,
            )

        return PerceptionRequirements(
            tasks=tasks,
            target_classes=target_classes,
            min_accuracy=min_accuracy,
            max_latency_ms=max_latency_ms,
            min_fps=min_fps,
        )

    def _show_task_options(self) -> None:
        """Display available task types."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Num", style="cyan")
        table.add_column("Task")
        table.add_column("Description", style="dim")

        task_descriptions = {
            TaskType.OBJECT_DETECTION: "Detect and localize objects with bounding boxes",
            TaskType.CLASSIFICATION: "Classify images into categories",
            TaskType.SEGMENTATION: "Pixel-wise semantic segmentation",
            TaskType.INSTANCE_SEGMENTATION: "Segment individual object instances",
            TaskType.POSE_ESTIMATION: "Estimate human body pose keypoints",
            TaskType.DEPTH_ESTIMATION: "Predict depth from monocular images",
            TaskType.FACE_DETECTION: "Detect and analyze faces",
            TaskType.HAND_TRACKING: "Track hand landmarks and gestures",
            TaskType.TRACKING: "Track objects across frames",
        }

        for i, task in enumerate(TaskType, 1):
            table.add_row(str(i), task.value, task_descriptions.get(task, ""))

        console.print("  Available tasks:")
        console.print(table)

    def _select_tasks(self) -> list[TaskType]:
        """Interactive task selection."""
        task_list = list(TaskType)

        selection = Prompt.ask(
            "  Select tasks (comma-separated numbers, e.g., 1,2)",
            default="1",
        )

        tasks = []
        for num in selection.split(","):
            try:
                idx = int(num.strip()) - 1
                if 0 <= idx < len(task_list):
                    tasks.append(task_list[idx])
            except ValueError:
                continue

        if not tasks:
            tasks = [TaskType.OBJECT_DETECTION]

        console.print(f"  Selected: {', '.join(t.value for t in tasks)}")
        return tasks

    def _get_hardware_requirements(self) -> HardwareRequirements:
        """Get hardware constraints."""
        console.print()
        console.print("[bold]Step 3: Hardware Constraints[/bold]")

        # Show execution targets
        console.print("  Execution targets:")
        for i, target in enumerate(ExecutionTarget, 1):
            console.print(f"    {i}. {target.value}")

        target_num = IntPrompt.ask(
            "  Select execution target",
            default=1,
        )
        target_list = list(ExecutionTarget)
        target_idx = max(0, min(target_num - 1, len(target_list) - 1))
        execution_target = target_list[target_idx]

        # Power constraint
        max_power = None
        if Confirm.ask("  Set power budget constraint?", default=False):
            max_power = FloatPrompt.ask(
                "    Maximum power (watts)",
                default=15.0,
            )

        # Memory constraint
        max_memory = None
        if Confirm.ask("  Set memory constraint?", default=False):
            max_memory = FloatPrompt.ask(
                "    Maximum memory (MB)",
                default=512.0,
            )

        # Parameter count constraint
        max_params = None
        if Confirm.ask("  Set model size constraint?", default=False):
            max_params = FloatPrompt.ask(
                "    Maximum parameters (millions)",
                default=10.0,
            )

        return HardwareRequirements(
            execution_target=execution_target,
            max_power_watts=max_power,
            max_memory_mb=max_memory,
            max_params_millions=max_params,
        )

    def _get_deployment_requirements(self) -> DeploymentRequirements:
        """Get deployment configuration."""
        console.print()
        console.print("[bold]Step 4: Deployment Configuration[/bold]")

        # Runtime
        runtime = Prompt.ask(
            "  Runtime (onnxruntime/tensorrt/openvino)",
            default="onnxruntime",
        )

        # Quantization
        quant_options = ["fp32", "fp16", "int8"]
        console.print("  Quantization options: " + ", ".join(quant_options))
        quantization = Prompt.ask(
            "  Quantization",
            default="fp16",
        )

        # Batch size
        batch_size = IntPrompt.ask(
            "  Batch size",
            default=1,
        )

        return DeploymentRequirements(
            runtime=runtime,
            quantization=quantization,
            batch_size=batch_size,
        )

    def _show_summary(self) -> None:
        """Display requirements summary."""
        console.print()
        console.print(
            Panel(
                self.requirements.summary(),
                title="[bold green]Requirements Summary[/bold green]",
                border_style="green",
            )
        )


def run_wizard() -> PipelineRequirements:
    """Convenience function to run the wizard.

    Returns:
        Configured PipelineRequirements
    """
    wizard = RequirementsWizard()
    return wizard.run()


def from_usecase(usecase_id: str) -> Optional[PipelineRequirements]:
    """Create requirements from an embodied-schemas use case.

    Args:
        usecase_id: Use case identifier

    Returns:
        PipelineRequirements or None if not found
    """
    try:
        from embodied_schemas import Registry

        registry = Registry()
        usecase = registry.get_usecase(usecase_id)

        if not usecase:
            console.print(f"[yellow]Use case '{usecase_id}' not found[/yellow]")
            return None

        # Map use case to requirements
        tasks = []
        if hasattr(usecase, "required_tasks"):
            task_map = {
                "detection": TaskType.OBJECT_DETECTION,
                "object_detection": TaskType.OBJECT_DETECTION,
                "classification": TaskType.CLASSIFICATION,
                "segmentation": TaskType.SEGMENTATION,
                "pose": TaskType.POSE_ESTIMATION,
                "pose_estimation": TaskType.POSE_ESTIMATION,
                "depth": TaskType.DEPTH_ESTIMATION,
                "depth_estimation": TaskType.DEPTH_ESTIMATION,
                "face": TaskType.FACE_DETECTION,
                "face_detection": TaskType.FACE_DETECTION,
                "hand": TaskType.HAND_TRACKING,
                "hand_tracking": TaskType.HAND_TRACKING,
                "tracking": TaskType.TRACKING,
            }
            for task in usecase.required_tasks:
                if task.lower() in task_map:
                    tasks.append(task_map[task.lower()])

        # Get constraints from use case
        min_accuracy = getattr(usecase, "min_accuracy", None)
        max_latency = getattr(usecase, "max_latency_ms", None)
        max_power = getattr(usecase, "max_power_watts", None)

        return PipelineRequirements(
            name=usecase_id.replace("_", "-"),
            description=getattr(usecase, "description", None),
            perception=PerceptionRequirements(
                tasks=tasks if tasks else [TaskType.OBJECT_DETECTION],
                min_accuracy=min_accuracy,
                max_latency_ms=max_latency,
            ),
            hardware=HardwareRequirements(
                max_power_watts=max_power,
            ),
            use_case=usecase_id,
        )

    except ImportError:
        console.print("[yellow]embodied-schemas not available[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Error loading use case: {e}[/red]")
        return None
