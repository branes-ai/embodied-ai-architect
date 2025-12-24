"""
Trajectory Evaluation for SoC Optimization Pipeline

This module analyzes execution trajectories to:
1. Verify correct tool usage
2. Detect anti-patterns
3. Compare against expected behavior
4. Generate actionable insights
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import Counter
import json


# ============================================================
# Trajectory Expectations
# ============================================================

@dataclass
class TrajectoryExpectation:
    """Defines what we expect from a successful optimization run."""
    
    # Tools that MUST be called
    required_tools: list = field(default_factory=lambda: [
        "rtl_lint",       # Must lint before anything
        "rtl_synthesis",  # Must synthesize to get metrics
        "simulation",     # Must validate functionality
    ])
    
    # Expected tool order (subsequence matching)
    expected_sequences: list = field(default_factory=lambda: [
        ["rtl_lint", "rtl_synthesis"],  # Lint before synth
        ["rtl_lint", "simulation"],     # Lint before sim
    ])
    
    # Tools that should NOT be called
    forbidden_tools: list = field(default_factory=lambda: [
        "web_search",    # Shouldn't search web for RTL
        "file_delete",   # Shouldn't delete files
    ])
    
    # Maximum retries
    max_retries_per_tool: int = 3
    
    # Maximum total tool calls
    max_total_tool_calls: int = 10


# Default expectation for our ALU optimization
ALU_OPTIMIZATION_EXPECTATION = TrajectoryExpectation(
    required_tools=["rtl_lint", "rtl_synthesis", "simulation"],
    expected_sequences=[
        ["rtl_lint", "rtl_synthesis"],
        ["rtl_lint", "simulation"],
    ],
    forbidden_tools=[],
    max_retries_per_tool=2,
    max_total_tool_calls=8
)


# ============================================================
# Trajectory Evaluator
# ============================================================

class TrajectoryEvaluator:
    """Evaluates execution trajectories against expectations."""
    
    def __init__(self, expectation: TrajectoryExpectation = None):
        self.expectation = expectation or ALU_OPTIMIZATION_EXPECTATION
    
    def evaluate(self, trajectory) -> dict:
        """
        Evaluate a trajectory and return detailed results.
        
        Args:
            trajectory: Trajectory object from pipeline execution
            
        Returns:
            Dictionary with evaluation results
        """
        
        # Extract tool sequence
        tool_sequence = trajectory.tool_sequence()
        
        results = {
            "tool_sequence": tool_sequence,
            "set_coverage": self._evaluate_set_coverage(tool_sequence),
            "sequence_coverage": self._evaluate_sequence_coverage(tool_sequence),
            "efficiency": self._evaluate_efficiency(trajectory),
            "anti_patterns": self._detect_anti_patterns(trajectory),
            "agent_analysis": self._analyze_by_agent(trajectory),
        }
        
        # Compute overall score
        results["summary"] = self._compute_summary(results)
        
        return results
    
    def _evaluate_set_coverage(self, tool_sequence: list) -> dict:
        """Check if required tools were called."""
        
        actual = set(tool_sequence)
        required = set(self.expectation.required_tools)
        forbidden = set(self.expectation.forbidden_tools)
        
        covered = required.intersection(actual)
        missing = required - actual
        violations = forbidden.intersection(actual)
        
        return {
            "required_tools": list(required),
            "covered": list(covered),
            "missing": list(missing),
            "forbidden_violations": list(violations),
            "coverage_pct": len(covered) / len(required) * 100 if required else 100,
            "passed": len(missing) == 0 and len(violations) == 0
        }
    
    def _evaluate_sequence_coverage(self, tool_sequence: list) -> dict:
        """Check if tools were called in expected order."""
        
        results = {
            "sequences": [],
            "all_passed": True
        }
        
        for expected_seq in self.expectation.expected_sequences:
            # Check if expected_seq is a subsequence of tool_sequence
            found = self._is_subsequence(expected_seq, tool_sequence)
            
            results["sequences"].append({
                "expected": expected_seq,
                "found": found
            })
            
            if not found:
                results["all_passed"] = False
        
        return results
    
    def _is_subsequence(self, needle: list, haystack: list) -> bool:
        """Check if needle appears in order within haystack."""
        it = iter(haystack)
        return all(item in it for item in needle)
    
    def _evaluate_efficiency(self, trajectory) -> dict:
        """Evaluate execution efficiency."""
        
        tool_calls = [s for s in trajectory.steps if s.action == "tool_call"]
        llm_calls = [s for s in trajectory.steps if s.action == "llm_generate"]
        
        # Count calls per tool
        tool_counts = Counter(s.tool_name for s in tool_calls if s.tool_name)
        
        # Check for excessive retries
        max_per_tool = max(tool_counts.values()) if tool_counts else 0
        
        # Time analysis
        total_time = sum(s.duration_ms for s in trajectory.steps)
        tool_time = sum(s.duration_ms for s in tool_calls)
        llm_time = sum(s.duration_ms for s in llm_calls)
        
        return {
            "total_steps": len(trajectory.steps),
            "tool_calls": len(tool_calls),
            "llm_calls": len(llm_calls),
            "tool_distribution": dict(tool_counts),
            "max_calls_per_tool": max_per_tool,
            "exceeds_retry_limit": max_per_tool > self.expectation.max_retries_per_tool,
            "exceeds_total_limit": len(tool_calls) > self.expectation.max_total_tool_calls,
            "time_breakdown": {
                "total_ms": total_time,
                "tool_ms": tool_time,
                "llm_ms": llm_time,
                "overhead_ms": total_time - tool_time - llm_time
            }
        }
    
    def _detect_anti_patterns(self, trajectory) -> list:
        """Detect problematic patterns."""
        
        anti_patterns = []
        
        # Anti-pattern 1: Repeated identical calls
        seen = {}  # (tool, input) -> indices
        for step in trajectory.steps:
            if step.action != "tool_call":
                continue
            key = (step.tool_name, step.input_summary)
            if key not in seen:
                seen[key] = []
            seen[key].append(step.index)
        
        for (tool, _), indices in seen.items():
            if len(indices) > 1:
                anti_patterns.append({
                    "type": "repeated_identical_call",
                    "tool": tool,
                    "count": len(indices),
                    "indices": indices,
                    "severity": "warning" if len(indices) == 2 else "error"
                })
        
        # Anti-pattern 2: Tool after failure without change
        prev = None
        for step in trajectory.steps:
            if step.action == "tool_call":
                if prev and prev.action == "tool_call":
                    if (prev.tool_name == step.tool_name and 
                        not prev.success and 
                        prev.input_summary == step.input_summary):
                        anti_patterns.append({
                            "type": "retry_without_change",
                            "tool": step.tool_name,
                            "indices": [prev.index, step.index],
                            "severity": "warning"
                        })
                prev = step
        
        # Anti-pattern 3: LLM generate without subsequent validation
        for i, step in enumerate(trajectory.steps):
            if step.action == "llm_generate" and step.agent == "Optimizer":
                # Check if followed by validation
                remaining = trajectory.steps[i+1:]
                has_validation = any(
                    s.action == "tool_call" and s.tool_name in ["simulation", "rtl_lint"]
                    for s in remaining
                )
                if not has_validation:
                    anti_patterns.append({
                        "type": "unvalidated_llm_output",
                        "index": step.index,
                        "severity": "error",
                        "message": "LLM-generated code not validated"
                    })
        
        # Anti-pattern 4: Synthesis before lint
        tool_sequence = trajectory.tool_sequence()
        try:
            lint_idx = tool_sequence.index("rtl_lint")
            synth_idx = tool_sequence.index("rtl_synthesis")
            if synth_idx < lint_idx:
                anti_patterns.append({
                    "type": "wrong_order",
                    "expected": "rtl_lint before rtl_synthesis",
                    "actual_order": tool_sequence,
                    "severity": "warning"
                })
        except ValueError:
            pass  # Tool not in sequence
        
        return anti_patterns
    
    def _analyze_by_agent(self, trajectory) -> dict:
        """Analyze trajectory by agent role."""
        
        by_agent = {}
        for step in trajectory.steps:
            if step.agent not in by_agent:
                by_agent[step.agent] = {
                    "steps": 0,
                    "tool_calls": 0,
                    "llm_calls": 0,
                    "failures": 0,
                    "total_time_ms": 0,
                    "tools_used": []
                }
            
            by_agent[step.agent]["steps"] += 1
            by_agent[step.agent]["total_time_ms"] += step.duration_ms
            
            if step.action == "tool_call":
                by_agent[step.agent]["tool_calls"] += 1
                if step.tool_name and step.tool_name not in by_agent[step.agent]["tools_used"]:
                    by_agent[step.agent]["tools_used"].append(step.tool_name)
            elif step.action == "llm_generate":
                by_agent[step.agent]["llm_calls"] += 1
            
            if not step.success:
                by_agent[step.agent]["failures"] += 1
        
        return by_agent
    
    def _compute_summary(self, results: dict) -> dict:
        """Compute overall evaluation summary."""
        
        score = 100.0
        issues = []
        
        # Penalize missing required tools
        set_cov = results["set_coverage"]
        if not set_cov["passed"]:
            missing = set_cov["missing"]
            score -= len(missing) * 20
            issues.append(f"Missing tools: {missing}")
        
        # Penalize forbidden tools
        if set_cov["forbidden_violations"]:
            score -= len(set_cov["forbidden_violations"]) * 25
            issues.append(f"Forbidden tools used: {set_cov['forbidden_violations']}")
        
        # Penalize sequence violations
        seq_cov = results["sequence_coverage"]
        if not seq_cov["all_passed"]:
            score -= 10
            failed_seqs = [s["expected"] for s in seq_cov["sequences"] if not s["found"]]
            issues.append(f"Sequence violations: {failed_seqs}")
        
        # Penalize efficiency issues
        eff = results["efficiency"]
        if eff["exceeds_retry_limit"]:
            score -= 5
            issues.append(f"Excessive retries: {eff['max_calls_per_tool']}")
        if eff["exceeds_total_limit"]:
            score -= 10
            issues.append(f"Too many tool calls: {eff['tool_calls']}")
        
        # Penalize anti-patterns
        for ap in results["anti_patterns"]:
            if ap["severity"] == "error":
                score -= 15
            else:
                score -= 5
            issues.append(f"{ap['type']}: {ap.get('tool', ap.get('message', ''))}")
        
        return {
            "score": max(0, min(100, score)),
            "grade": self._score_to_grade(score),
            "passed": score >= 70,
            "issues": issues
        }
    
    def _score_to_grade(self, score: float) -> str:
        if score >= 90: return "A"
        if score >= 80: return "B"
        if score >= 70: return "C"
        if score >= 60: return "D"
        return "F"
    
    def print_report(self, results: dict):
        """Print a human-readable evaluation report."""
        
        print("\n" + "="*60)
        print("TRAJECTORY EVALUATION REPORT")
        print("="*60)
        
        # Summary
        summary = results["summary"]
        print(f"\nOverall Score: {summary['score']:.0f}/100 (Grade: {summary['grade']})")
        print(f"Status: {'PASSED ✓' if summary['passed'] else 'FAILED ✗'}")
        
        # Tool coverage
        print("\n--- Tool Coverage ---")
        set_cov = results["set_coverage"]
        print(f"Required: {set_cov['required_tools']}")
        print(f"Covered:  {set_cov['covered']}")
        if set_cov['missing']:
            print(f"Missing:  {set_cov['missing']} ✗")
        if set_cov['forbidden_violations']:
            print(f"Forbidden used: {set_cov['forbidden_violations']} ✗")
        
        # Sequence coverage
        print("\n--- Sequence Coverage ---")
        for seq in results["sequence_coverage"]["sequences"]:
            status = "✓" if seq["found"] else "✗"
            print(f"  {seq['expected']}: {status}")
        
        # Efficiency
        print("\n--- Efficiency ---")
        eff = results["efficiency"]
        print(f"Total steps: {eff['total_steps']}")
        print(f"Tool calls: {eff['tool_calls']}")
        print(f"LLM calls: {eff['llm_calls']}")
        print(f"Tool distribution: {eff['tool_distribution']}")
        print(f"Time breakdown: {eff['time_breakdown']['total_ms']:.0f}ms total")
        
        # Anti-patterns
        if results["anti_patterns"]:
            print("\n--- Anti-Patterns Detected ---")
            for ap in results["anti_patterns"]:
                severity = "⚠️" if ap["severity"] == "warning" else "❌"
                print(f"  {severity} {ap['type']}: {ap.get('tool', ap.get('message', ''))}")
        
        # Issues
        if summary["issues"]:
            print("\n--- Issues ---")
            for issue in summary["issues"]:
                print(f"  • {issue}")
        
        # Agent analysis
        print("\n--- By Agent ---")
        for agent, stats in results["agent_analysis"].items():
            print(f"  {agent}:")
            print(f"    Steps: {stats['steps']}, Tools: {stats['tool_calls']}, LLM: {stats['llm_calls']}")
            print(f"    Failures: {stats['failures']}, Time: {stats['total_time_ms']:.0f}ms")


# ============================================================
# Comparison Utilities
# ============================================================

def compare_trajectories(traj1_results: dict, traj2_results: dict, 
                         name1: str = "Run 1", name2: str = "Run 2") -> dict:
    """Compare two trajectory evaluations."""
    
    comparison = {
        "scores": {
            name1: traj1_results["summary"]["score"],
            name2: traj2_results["summary"]["score"]
        },
        "better": name1 if traj1_results["summary"]["score"] > traj2_results["summary"]["score"] else name2,
        "differences": []
    }
    
    # Compare tool coverage
    set1 = set(traj1_results["set_coverage"]["covered"])
    set2 = set(traj2_results["set_coverage"]["covered"])
    
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1
    
    if only_in_1:
        comparison["differences"].append(f"{name1} used extra tools: {only_in_1}")
    if only_in_2:
        comparison["differences"].append(f"{name2} used extra tools: {only_in_2}")
    
    # Compare efficiency
    eff1 = traj1_results["efficiency"]
    eff2 = traj2_results["efficiency"]
    
    if eff1["tool_calls"] != eff2["tool_calls"]:
        comparison["differences"].append(
            f"Tool call difference: {name1}={eff1['tool_calls']}, {name2}={eff2['tool_calls']}"
        )
    
    # Compare anti-patterns
    ap1_types = set(ap["type"] for ap in traj1_results["anti_patterns"])
    ap2_types = set(ap["type"] for ap in traj2_results["anti_patterns"])
    
    only_in_1 = ap1_types - ap2_types
    only_in_2 = ap2_types - ap1_types
    
    if only_in_1:
        comparison["differences"].append(f"{name1} unique anti-patterns: {only_in_1}")
    if only_in_2:
        comparison["differences"].append(f"{name2} unique anti-patterns: {only_in_2}")
    
    return comparison


# ============================================================
# Demo
# ============================================================

def demo():
    """Demonstrate trajectory evaluation."""
    
    # Run the pipeline
    from pipeline import OptimizationPipeline
    from pathlib import Path
    
    designs_dir = Path(__file__).parent / "designs"
    original_rtl = (designs_dir / "alu_baseline.v").read_text()
    testbench = (designs_dir / "alu_tb.v").read_text()
    
    print("Running pipelines and collecting trajectories...")
    
    # Run 1: Buggy optimizer
    pipeline1 = OptimizationPipeline(work_dir=Path("./eval_buggy"), use_buggy_optimizer=True)
    result1 = pipeline1.run(original_rtl, testbench)
    
    # Run 2: Correct optimizer  
    pipeline2 = OptimizationPipeline(work_dir=Path("./eval_correct"), use_buggy_optimizer=False)
    result2 = pipeline2.run(original_rtl, testbench)
    
    # Evaluate trajectories
    evaluator = TrajectoryEvaluator()
    
    print("\n" + "#"*60)
    print("# EVALUATING BUGGY RUN")
    print("#"*60)
    eval1 = evaluator.evaluate(result1.trajectory)
    evaluator.print_report(eval1)
    
    print("\n" + "#"*60)
    print("# EVALUATING CORRECT RUN")
    print("#"*60)
    eval2 = evaluator.evaluate(result2.trajectory)
    evaluator.print_report(eval2)
    
    # Compare
    print("\n" + "#"*60)
    print("# COMPARISON")
    print("#"*60)
    comparison = compare_trajectories(eval1, eval2, "Buggy", "Correct")
    print(f"\nScores: Buggy={comparison['scores']['Buggy']:.0f}, Correct={comparison['scores']['Correct']:.0f}")
    print(f"Better trajectory: {comparison['better']}")
    if comparison["differences"]:
        print("\nDifferences:")
        for diff in comparison["differences"]:
            print(f"  • {diff}")
    
    return eval1, eval2, comparison


if __name__ == "__main__":
    demo()
