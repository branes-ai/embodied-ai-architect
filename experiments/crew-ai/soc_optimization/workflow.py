#!/usr/bin/env python3
"""
Complete Workflow Demonstration: SoC Optimization Agent Evaluation

This script demonstrates a FULL evaluation harness including:
1. Running the optimization pipeline
2. Trajectory collection and analysis
3. Ground-truth validation
4. Comparative analysis

Run with: python workflow.py
"""

import sys
from pathlib import Path
from dataclasses import asdict
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tools import RTLLintTool, RTLSynthesisTool, SimulationTool
from pipeline import OptimizationPipeline, OptimizationResult
from trajectory_eval import TrajectoryEvaluator, compare_trajectories


def print_section(title: str):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def run_workflow():
    """Run the complete demonstration."""
    
    print_section("SoC OPTIMIZATION AGENT EVALUATION DEMO")
    
    print("""
This agentic optimization workflow demo shows how to evaluate 
an LLM-based SoC optimization system.

We will:
1. Run the optimizer with BUGGY LLM output (emulating common failures)
2. Run the optimizer with CORRECT LLM output (ideal case)
3. Analyze trajectories to trace what happened
4. Compare the two runs
5. Draw conclusions about agent architecture

The key insight: LLMs can be USEFUL but UNRELIABLE for hardware design.
Proper evaluation catches failures that would otherwise ship bugs.
""")
    
    # Load designs
    designs_dir = Path(__file__).parent / "designs"
    original_rtl = (designs_dir / "alu_baseline.v").read_text()
    testbench = (designs_dir / "alu_tb.v").read_text()
    
    print_section("PHASE 1: Run Optimization with BUGGY LLM")
    
    print("""
Emulating a common LLM failure: The optimizer will:
- Correctly identify optimization opportunities ✓
- Correctly implement add/sub resource sharing ✓
- INCORRECTLY "simplify" the barrel shifter ✗
  (Changes shift-by-N to shift-by-1, breaking functionality)
""")
    
    pipeline_buggy = OptimizationPipeline(
        work_dir=Path("./demo_buggy"),
        use_buggy_optimizer=True
    )
    result_buggy = pipeline_buggy.run(original_rtl, testbench)
    
    print_section("PHASE 2: Run Optimization with CORRECT LLM")
    
    print("""
Simulating ideal behavior: The optimizer will:
- Correctly identify optimization opportunities ✓
- Correctly implement add/sub resource sharing ✓
- Preserve shifter functionality ✓
""")
    
    pipeline_correct = OptimizationPipeline(
        work_dir=Path("./demo_correct"),
        use_buggy_optimizer=False
    )
    result_correct = pipeline_correct.run(original_rtl, testbench)
    
    print_section("PHASE 3: Trajectory Analysis")
    
    evaluator = TrajectoryEvaluator()
    
    print("\n--- Buggy Run Trajectory ---")
    eval_buggy = evaluator.evaluate(result_buggy.trajectory)
    evaluator.print_report(eval_buggy)
    
    print("\n--- Correct Run Trajectory ---")
    eval_correct = evaluator.evaluate(result_correct.trajectory)
    evaluator.print_report(eval_correct)
    
    print_section("PHASE 4: Comparison")
    
    comparison = compare_trajectories(eval_buggy, eval_correct, "Buggy", "Correct")
    
    print(f"""
Trajectory Scores:
  Buggy run:   {comparison['scores']['Buggy']:.0f}/100
  Correct run: {comparison['scores']['Correct']:.0f}/100

Note: Both trajectories may score SIMILARLY because they followed
the same process. The difference is in the OUTCOME, not the TRAJECTORY.

This highlights an important distinction:
- Trajectory evaluation catches PROCESS issues (wrong tools, bad order)
- Outcome evaluation catches RESULT issues (broken functionality)

You need BOTH for comprehensive evaluation.
""")
    
    print_section("PHASE 5: Outcome Comparison")
    
    print(f"""
BUGGY RUN:
  Validation: {'PASSED ✓' if result_buggy.validation_passed else 'FAILED ✗'}
  Original cells: {result_buggy.original_metrics.get('area_cells', 'N/A')}
  Optimized cells: {result_buggy.optimized_metrics.get('area_cells', 'N/A') if result_buggy.optimized_metrics else 'N/A'}

CORRECT RUN:
  Validation: {'PASSED ✓' if result_correct.validation_passed else 'FAILED ✗'}
  Original cells: {result_correct.original_metrics.get('area_cells', 'N/A')}
  Optimized cells: {result_correct.optimized_metrics.get('area_cells', 'N/A') if result_correct.optimized_metrics else 'N/A'}
""")
    
    print_section("KEY TAKEAWAYS")
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                WHAT THIS WORKFLOW DEMO SHOWS                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WHERE LLMS ARE HELPFUL:                                            │
│  o Identifying optimization opportunities in code                   │
│  o Suggesting common transformations (resource sharing)             │
│  o Generating syntactically valid code                              │
│  o Following structured workflows                                   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WHERE LLMS FAIL:                                                   │
│  x Maintaining semantic correctness during transformations          │
│  x Preserving subtle behavioral requirements                        │
│  x Knowing when "simplification" changes functionality              │
│  x Reasoning about all edge cases                                   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WHY THE AGENT ARCHITECTURE MATTERS:                                │
│                                                                     │
│  The pipeline: Analyze → Optimize → Validate → Score                │
│                                                                     │
│  1. ANALYZE: LLM examines code (usually good at this)               │
│  2. OPTIMIZE: LLM generates new code (UNRELIABLE)                   │
│  3. VALIDATE: Ground-truth simulation CATCHES LLM BUGS              │
│  4. SCORE: Real metrics from EDA tools (objective)                  │
│                                                                     │
│  Without step 3, the buggy shifter would have poluted the design!   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  EVALUATION STRATEGY:                                               │
│                                                                     │
│  1. TRAJECTORY EVALUATION - Did the agent follow correct process?   │
│     • Required tools called                                         │
│     • Correct order                                                 │
│     • No anti-patterns                                              │
│                                                                     │
│  2. OUTCOME EVALUATION - Did it produce correct results?            │
│     • Functional simulation (ground truth)                          │
│     • PPA metrics (real EDA tools)                                  │
│     • Constraint satisfaction                                       │
│                                                                     │
│  3. COMPARATIVE EVALUATION - How does it compare to baseline?       │
│     • Area reduction                                                │
│     • No timing regression                                          │
│     • No power increase                                             │
│                                                                     │
│  You need ALL THREE to trust an LLM-based optimization system.      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print_section("PRACTICAL RECOMMENDATIONS")
    
    print("""
For your SoC optimization system:

1. NEVER TRUST LLM OUTPUT WITHOUT VALIDATION
   - Always run functional Verification and Validation
   - Always run synthesis to verify synthesizability
   - Consider formal verification for critical paths

2. DESIGN FOR DEFENSE IN DEPTH
   - Multiple validation stages
   - Human review for non-trivial changes
   - Incremental changes (don't let LLM rewrite everything)

3. TRACK TRAJECTORIES
   - Log every tool call
   - Monitor for anti-patterns
   - Detect regressions early

4. BUILD A GROUND-TRUTH CORPUS
   - Known-good optimizations
   - Edge cases that trip up LLMs
   - Regression tests

5. SET CLEAR THRESHOLDS
   - Minimum validation pass rate
   - Maximum acceptable area increase
   - Minimum timing slack

The agent architecture (CrewAI, etc.) provides ORCHESTRATION.
But correctness comes from VALIDATION and GROUND TRUTH.
""")
    
    return {
        "buggy": {"result": result_buggy, "eval": eval_buggy},
        "correct": {"result": result_correct, "eval": eval_correct},
        "comparison": comparison
    }


if __name__ == "__main__":
    results = run_workflow()
    
    # Save results for later analysis
    output_path = Path("./demo_results.json")
    
    # Convert to serializable format
    output = {
        "buggy": {
            "validation_passed": results["buggy"]["result"].validation_passed,
            "trajectory_score": results["buggy"]["eval"]["summary"]["score"],
            "original_cells": results["buggy"]["result"].original_metrics.get("area_cells"),
            "optimized_cells": results["buggy"]["result"].optimized_metrics.get("area_cells") if results["buggy"]["result"].optimized_metrics else None,
        },
        "correct": {
            "validation_passed": results["correct"]["result"].validation_passed,
            "trajectory_score": results["correct"]["eval"]["summary"]["score"],
            "original_cells": results["correct"]["result"].original_metrics.get("area_cells"),
            "optimized_cells": results["correct"]["result"].optimized_metrics.get("area_cells") if results["correct"]["result"].optimized_metrics else None,
        },
        "comparison": results["comparison"]
    }
    
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")
