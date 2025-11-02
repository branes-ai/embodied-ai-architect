"""Benchmarking commands."""

import json

import click
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from embodied_ai_architect.agents.benchmark import BenchmarkAgent
from embodied_ai_architect.agents.benchmark.backends.local_cpu import LocalCPUBackend

console = Console()


@click.group()
def benchmark():
    """Run performance benchmarks.

    \b
    Examples:
      # Benchmark on local CPU
      embodied-ai benchmark run my_model.pt

      # Benchmark on specific backend
      embodied-ai benchmark run my_model.pt --backend kubernetes

      # List available backends
      embodied-ai benchmark list
    """
    pass


@benchmark.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--backend",
    default="local_cpu",
    help="Benchmark backend (local_cpu, remote_ssh, kubernetes)",
)
@click.option("--input-shape", help="Model input shape (e.g., 1,3,224,224)")
@click.option("--iterations", default=100, help="Number of benchmark iterations")
@click.option("--warmup", default=10, help="Number of warmup iterations")
@click.pass_context
def run(ctx, model_path, backend, input_shape, iterations, warmup):
    """Run benchmark on a model."""
    json_output = ctx.obj.get("json", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        # Load model
        if not quiet and not json_output:
            console.print(f"\n[cyan]Loading model:[/cyan] {model_path}")

        model = torch.load(model_path, weights_only=False)
        model.eval()

        # Parse input shape if provided
        if input_shape:
            input_shape_tuple = tuple(map(int, input_shape.split(",")))
        else:
            # Default shape
            input_shape_tuple = (1, 3, 224, 224)

        # Create backend instance
        if backend == "local_cpu":
            backend_instance = LocalCPUBackend()
        else:
            # For other backends, we'd need to load them here
            console.print(f"[yellow]⚠[/yellow] Backend '{backend}' not yet implemented in CLI")
            console.print("[dim]Using local_cpu backend instead[/dim]")
            backend_instance = LocalCPUBackend()

        # Run benchmark
        agent = BenchmarkAgent(backends=[backend_instance])

        if not quiet and not json_output:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"[cyan]Benchmarking on {backend}...", total=None)
                result = agent.execute(
                    {
                        "model": model,
                        "input_shape": input_shape_tuple,
                        "iterations": iterations,
                        "warmup_iterations": warmup,
                    }
                )
                progress.update(task, completed=1)
        else:
            result = agent.execute(
                {
                    "model": model,
                    "input_shape": input_shape_tuple,
                    "iterations": iterations,
                    "warmup_iterations": warmup,
                }
            )

        if not result.success:
            raise Exception(result.error)

        # Output results
        if json_output:
            click.echo(json.dumps(result.data, indent=2))
        else:
            console.print(f"\n[bold green]✓[/bold green] Benchmark complete\n")

            # Get benchmark data for the backend
            benchmarks = result.data.get("benchmarks", {})
            backend_result = benchmarks.get(backend_instance.name, {})

            console.print("[bold]Benchmark Results[/bold]")
            console.print(f"  Backend: {backend_result.get('device', backend)}")
            console.print(f"  Iterations: {backend_result.get('iterations', iterations)}")
            console.print(f"  Mean Latency: {backend_result.get('mean_latency_ms', 0):.3f} ms")
            console.print(f"  Std Latency: {backend_result.get('std_latency_ms', 0):.3f} ms")
            console.print(f"  Min Latency: {backend_result.get('min_latency_ms', 0):.3f} ms")
            console.print(f"  Max Latency: {backend_result.get('max_latency_ms', 0):.3f} ms")
            console.print(
                f"  Throughput: {backend_result.get('throughput_samples_per_sec', 0):.2f} samples/sec"
            )

    except FileNotFoundError:
        console.print(f"\n[bold red]❌ Error:[/bold red] Model file not found: {model_path}")
        ctx.exit(1)
    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        ctx.exit(1)


@benchmark.command()
@click.pass_context
def list(ctx):
    """List available benchmark backends."""
    json_output = ctx.obj.get("json", False)

    backends = [
        {
            "name": "local_cpu",
            "description": "Run benchmarks on local CPU",
            "available": True,
            "requirements": "None",
        },
        {
            "name": "remote_ssh",
            "description": "Run benchmarks on remote machine via SSH",
            "available": "paramiko" in dir(),
            "requirements": "pip install embodied-ai-architect[remote]",
        },
        {
            "name": "kubernetes",
            "description": "Run benchmarks on Kubernetes cluster",
            "available": "kubernetes" in dir(),
            "requirements": "pip install embodied-ai-architect[kubernetes]",
        },
    ]

    if json_output:
        click.echo(json.dumps({"backends": backends}, indent=2))
    else:
        table = Table(title="Available Backends", show_header=True)
        table.add_column("Backend", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Available", style="green")
        table.add_column("Requirements", style="yellow")

        for backend in backends:
            table.add_row(
                backend["name"],
                backend["description"],
                "✓" if backend["available"] else "✗",
                backend["requirements"],
            )

        console.print(table)
