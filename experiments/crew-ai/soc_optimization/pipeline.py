"""
SoC Optimization Agents and Workflow

This module defines the multi-agent system for RTL optimization.
It demonstrates both the power and limitations of LLM-based design agents.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from enum import Enum

# Import tools
from tools import RTLLintTool, RTLSynthesisTool, SimulationTool


# ============================================================
# Trajectory Tracking
# ============================================================

@dataclass
class TrajectoryStep:
    """A single step in the agent's execution."""
    index: int
    timestamp: float
    agent: str
    action: str  # "tool_call", "llm_generate", "decision"
    tool_name: Optional[str]
    input_summary: str
    output_summary: str
    success: bool
    duration_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    """Complete execution trajectory."""
    steps: list = field(default_factory=list)
    start_time: float = 0
    
    def add_step(self, agent: str, action: str, tool_name: str,
                 input_summary: str, output_summary: str, 
                 success: bool, duration_ms: float, metadata: dict = None):
        step = TrajectoryStep(
            index=len(self.steps),
            timestamp=time.time() - self.start_time,
            agent=agent,
            action=action,
            tool_name=tool_name,
            input_summary=input_summary[:200],
            output_summary=output_summary[:200],
            success=success,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        self.steps.append(step)
        return step
    
    def tool_sequence(self) -> list:
        """Get ordered list of tools called."""
        return [s.tool_name for s in self.steps if s.action == "tool_call" and s.tool_name]
    
    def print_summary(self):
        """Print human-readable summary."""
        print(f"\n{'='*60}")
        print("TRAJECTORY SUMMARY")
        print(f"{'='*60}")
        print(f"Total steps: {len(self.steps)}")
        total_time = sum(s.duration_ms for s in self.steps)
        print(f"Total time: {total_time:.0f}ms")
        print()
        
        for step in self.steps:
            status = "✓" if step.success else "✗"
            tool_info = f"[{step.tool_name}]" if step.tool_name else ""
            print(f"  {step.index:2d}. [{step.agent:15s}] {step.action:12s} {tool_info:20s} {status}")


# ============================================================
# Agent Definitions (Simulated - no actual LLM calls)
# ============================================================

class AgentRole(Enum):
    ANALYZER = "analyzer"
    OPTIMIZER = "optimizer"
    VALIDATOR = "validator"
    SCORER = "scorer"


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    role: AgentRole
    name: str
    goal: str
    allowed_tools: list


# Define our agents
AGENTS = {
    AgentRole.ANALYZER: AgentConfig(
        role=AgentRole.ANALYZER,
        name="Design Analyzer",
        goal="Analyze RTL structure and identify optimization opportunities",
        allowed_tools=["rtl_lint"]
    ),
    AgentRole.OPTIMIZER: AgentConfig(
        role=AgentRole.OPTIMIZER,
        name="Design Optimizer", 
        goal="Generate optimized RTL based on analysis",
        allowed_tools=["rtl_lint", "rtl_synthesis"]
    ),
    AgentRole.VALIDATOR: AgentConfig(
        role=AgentRole.VALIDATOR,
        name="Design Validator",
        goal="Verify optimized design is functionally correct",
        allowed_tools=["simulation", "rtl_lint"]
    ),
    AgentRole.SCORER: AgentConfig(
        role=AgentRole.SCORER,
        name="PPA Scorer",
        goal="Compare PPA metrics between original and optimized designs",
        allowed_tools=["rtl_synthesis"]
    ),
}


# ============================================================
# Mock LLM Responses (Simulating what an LLM would generate)
# ============================================================

def mock_analyzer_response(lint_result: dict) -> dict:
    """
    Simulates what an LLM analyzer would produce.
    
    In a real system, this would be an LLM call with the lint results.
    Here we show both GOOD and BAD patterns that LLMs exhibit.
    """
    
    analysis = {
        "module_name": lint_result.get("module_name", "unknown"),
        "observations": [],
        "optimization_opportunities": [],
        "risks": []
    }
    
    # GOOD: LLMs can identify structural patterns
    if lint_result.get("ports"):
        analysis["observations"].append(
            f"Module has {len(lint_result['ports'])} ports"
        )
    
    if lint_result.get("parameters"):
        analysis["observations"].append(
            f"Parameterized design with {len(lint_result['parameters'])} parameters"
        )
        analysis["optimization_opportunities"].append(
            "Consider parameter tuning for specific use case"
        )
    
    # GOOD: LLMs can suggest common optimizations
    analysis["optimization_opportunities"].extend([
        "Resource sharing between add/subtract operations",
        "Combine comparison operations with subtraction",
        "Consider async reset for area reduction",
    ])
    
    # BAD: LLMs often suggest vague or inapplicable optimizations
    analysis["optimization_opportunities"].extend([
        "Use advanced synthesis directives",  # Too vague
        "Pipeline for higher frequency",       # May not be applicable
    ])
    
    # GOOD: LLMs can identify risks
    analysis["risks"].extend([
        "Changing reset behavior may affect system integration",
        "Resource sharing may increase critical path"
    ])
    
    return analysis


def mock_optimizer_response(original_rtl: str, analysis: dict) -> str:
    """
    Simulates what an LLM optimizer would produce.
    
    This shows BOTH successful optimizations AND common LLM failures.
    We'll generate a partially optimized design with some issues.
    """
    
    # An LLM might produce this optimized version
    # NOTE: This is intentionally imperfect to demonstrate LLM limitations
    
    optimized_rtl = '''// Optimized ALU - generated by LLM optimizer
// Attempted optimizations:
// 1. Shared adder for add/sub (CORRECT)
// 2. Reused subtraction for comparison (CORRECT)
// 3. Async reset (CORRECT but may cause issues)
// 4. Removed barrel shifter (INCORRECT - changes functionality!)

module alu_optimized #(
    parameter WIDTH = 8
)(
    input  wire                clk,
    input  wire                rst,
    input  wire [WIDTH-1:0]    a,
    input  wire [WIDTH-1:0]    b,
    input  wire [3:0]          op,
    input  wire                valid_in,
    output reg  [WIDTH-1:0]    result,
    output reg                 zero,
    output reg                 overflow,
    output reg                 valid_out
);

    // Operation codes
    localparam OP_ADD  = 4'b0000;
    localparam OP_SUB  = 4'b0001;
    localparam OP_AND  = 4'b0010;
    localparam OP_OR   = 4'b0011;
    localparam OP_XOR  = 4'b0100;
    localparam OP_SHL  = 4'b0101;
    localparam OP_SHR  = 4'b0110;
    localparam OP_LT   = 4'b0111;
    localparam OP_EQ   = 4'b1000;
    localparam OP_PASS = 4'b1001;

    // OPTIMIZATION 1: Single adder with carry-in for add/sub
    wire [WIDTH:0] addsub_result;
    wire subtract = (op == OP_SUB) || (op == OP_LT) || (op == OP_EQ);
    wire [WIDTH-1:0] b_mux = subtract ? ~b : b;
    assign addsub_result = {1'b0, a} + {1'b0, b_mux} + {{WIDTH{1'b0}}, subtract};
    
    // OPTIMIZATION 2: Reuse subtraction for comparisons
    wire lt_result = addsub_result[WIDTH];  // Sign bit of subtraction
    wire eq_result = (addsub_result[WIDTH-1:0] == {WIDTH{1'b0}});
    
    // Result computation
    reg [WIDTH-1:0] result_comb;
    reg overflow_comb;
    
    always @(*) begin
        result_comb = {WIDTH{1'b0}};
        overflow_comb = 1'b0;
        
        case (op)
            OP_ADD: begin
                result_comb = addsub_result[WIDTH-1:0];
                overflow_comb = addsub_result[WIDTH];
            end
            
            OP_SUB: begin
                result_comb = addsub_result[WIDTH-1:0];
                overflow_comb = addsub_result[WIDTH];
            end
            
            OP_AND: result_comb = a & b;
            OP_OR:  result_comb = a | b;
            OP_XOR: result_comb = a ^ b;
            
            // BUG: LLM "simplified" the shifter incorrectly!
            // Original: shift by b[2:0] (0-7 positions)
            // Broken: always shifts by 1
            OP_SHL: result_comb = a << 1;  // BUG: should be a << b[2:0]
            OP_SHR: result_comb = a >> 1;  // BUG: should be a >> b[2:0]
            
            OP_LT: result_comb = {{(WIDTH-1){1'b0}}, lt_result};
            OP_EQ: result_comb = {{(WIDTH-1){1'b0}}, eq_result};
            
            OP_PASS: result_comb = a;
            
            default: result_comb = {WIDTH{1'b0}};
        endcase
    end
    
    // OPTIMIZATION 3: Async reset (saves area but may cause issues)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            result    <= {WIDTH{1'b0}};
            zero      <= 1'b0;
            overflow  <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            result    <= result_comb;
            zero      <= (result_comb == {WIDTH{1'b0}});
            overflow  <= overflow_comb;
            valid_out <= valid_in;
        end
    end

endmodule
'''
    
    return optimized_rtl


def mock_optimizer_response_correct(original_rtl: str, analysis: dict) -> str:
    """
    A CORRECT optimization that an LLM might produce on a good day.
    """
    
    optimized_rtl = '''// Optimized ALU - correct version
// Optimizations applied:
// 1. Shared adder for add/sub using carry-in
// 2. Reused subtraction result for LT/EQ comparisons
// 3. Kept shifter functionality intact

module alu_optimized #(
    parameter WIDTH = 8
)(
    input  wire                clk,
    input  wire                rst,
    input  wire [WIDTH-1:0]    a,
    input  wire [WIDTH-1:0]    b,
    input  wire [3:0]          op,
    input  wire                valid_in,
    output reg  [WIDTH-1:0]    result,
    output reg                 zero,
    output reg                 overflow,
    output reg                 valid_out
);

    localparam OP_ADD  = 4'b0000;
    localparam OP_SUB  = 4'b0001;
    localparam OP_AND  = 4'b0010;
    localparam OP_OR   = 4'b0011;
    localparam OP_XOR  = 4'b0100;
    localparam OP_SHL  = 4'b0101;
    localparam OP_SHR  = 4'b0110;
    localparam OP_LT   = 4'b0111;
    localparam OP_EQ   = 4'b1000;
    localparam OP_PASS = 4'b1001;

    // Shared adder/subtractor
    wire subtract = (op == OP_SUB) || (op == OP_LT) || (op == OP_EQ);
    wire [WIDTH-1:0] b_operand = subtract ? ~b : b;
    wire [WIDTH:0] addsub_result = {1'b0, a} + {1'b0, b_operand} + {{WIDTH{1'b0}}, subtract};
    
    // Comparisons from subtraction
    wire signed [WIDTH-1:0] a_signed = a;
    wire signed [WIDTH-1:0] b_signed = b;
    wire lt_result = (a_signed < b_signed);
    wire eq_result = (addsub_result[WIDTH-1:0] == {WIDTH{1'b0}});
    
    reg [WIDTH-1:0] result_comb;
    reg overflow_comb;
    
    always @(*) begin
        result_comb = {WIDTH{1'b0}};
        overflow_comb = 1'b0;
        
        case (op)
            OP_ADD: begin
                result_comb = addsub_result[WIDTH-1:0];
                overflow_comb = addsub_result[WIDTH];
            end
            OP_SUB: begin
                result_comb = addsub_result[WIDTH-1:0];
                overflow_comb = addsub_result[WIDTH];
            end
            OP_AND: result_comb = a & b;
            OP_OR:  result_comb = a | b;
            OP_XOR: result_comb = a ^ b;
            OP_SHL: result_comb = a << b[2:0];  // Preserved correctly
            OP_SHR: result_comb = a >> b[2:0];  // Preserved correctly
            OP_LT:  result_comb = {{(WIDTH-1){1'b0}}, lt_result};
            OP_EQ:  result_comb = {{(WIDTH-1){1'b0}}, eq_result};
            OP_PASS: result_comb = a;
            default: result_comb = {WIDTH{1'b0}};
        endcase
    end
    
    always @(posedge clk) begin
        if (rst) begin
            result    <= {WIDTH{1'b0}};
            zero      <= 1'b0;
            overflow  <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            result    <= result_comb;
            zero      <= (result_comb == {WIDTH{1'b0}});
            overflow  <= overflow_comb;
            valid_out <= valid_in;
        end
    end

endmodule
'''
    
    return optimized_rtl


# ============================================================
# Main Optimization Flow
# ============================================================

@dataclass
class OptimizationResult:
    """Result of the optimization pipeline."""
    success: bool
    original_metrics: dict
    optimized_metrics: dict = None
    improvement: dict = None
    validation_passed: bool = False
    trajectory: Trajectory = None
    error: str = None
    
    def summary(self) -> str:
        lines = [
            "\n" + "="*60,
            "OPTIMIZATION RESULT",
            "="*60,
        ]
        
        if not self.success:
            lines.append(f"FAILED: {self.error}")
            return "\n".join(lines)
        
        lines.append(f"Validation: {'PASSED ✓' if self.validation_passed else 'FAILED ✗'}")
        lines.append("")
        lines.append("Metrics Comparison:")
        lines.append(f"  Original cells:  {self.original_metrics.get('area_cells', 'N/A')}")
        if self.optimized_metrics:
            lines.append(f"  Optimized cells: {self.optimized_metrics.get('area_cells', 'N/A')}")
        if self.improvement:
            lines.append(f"  Area reduction:  {self.improvement.get('area_reduction_pct', 0):.1f}%")
        
        return "\n".join(lines)


class OptimizationPipeline:
    """
    Main optimization pipeline.
    
    This demonstrates the full agent workflow with trajectory tracking.
    """
    
    def __init__(self, work_dir: Path = None, use_buggy_optimizer: bool = True):
        self.work_dir = work_dir or Path("./optimization_runs")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tools
        self.lint_tool = RTLLintTool(self.work_dir / "lint")
        self.synth_tool = RTLSynthesisTool(self.work_dir / "synth")
        self.sim_tool = SimulationTool(self.work_dir / "sim")
        
        # Configuration
        self.use_buggy_optimizer = use_buggy_optimizer  # Demonstrate LLM failures
        
        # Trajectory
        self.trajectory = Trajectory()
    
    def run(self, original_rtl: str, testbench: str, top_module: str = "alu_baseline") -> OptimizationResult:
        """Run the full optimization pipeline."""
        
        self.trajectory = Trajectory()
        self.trajectory.start_time = time.time()
        
        result = OptimizationResult(
            success=False,
            original_metrics={},
            trajectory=self.trajectory
        )
        
        try:
            # ===== Phase 1: Analyze Original Design =====
            print("\n[Phase 1] Analyzing original design...")
            
            # Tool call: Lint
            start = time.time()
            lint_result = self.lint_tool.run(original_rtl)
            duration = (time.time() - start) * 1000
            
            self.trajectory.add_step(
                agent="Analyzer",
                action="tool_call",
                tool_name="rtl_lint",
                input_summary=f"Lint {top_module}",
                output_summary=f"Success={bool(lint_result.get('module_name'))}, Module={lint_result.get('module_name')}",
                success=bool(lint_result.get('module_name')),
                duration_ms=duration
            )
            
            # Continue even if lint has warnings (common case)
            if lint_result.get('errors') and not lint_result.get('module_name'):
                result.error = f"Lint failed: {lint_result.get('errors')}"
                return result
            
            # LLM call: Analyze
            start = time.time()
            analysis = mock_analyzer_response(lint_result)
            duration = (time.time() - start) * 1000
            
            self.trajectory.add_step(
                agent="Analyzer",
                action="llm_generate",
                tool_name=None,
                input_summary="Generate analysis from lint results",
                output_summary=f"Found {len(analysis['optimization_opportunities'])} opportunities",
                success=True,
                duration_ms=duration
            )
            
            print(f"  Found {len(analysis['optimization_opportunities'])} optimization opportunities")
            
            # ===== Phase 2: Synthesize Original (Baseline) =====
            print("\n[Phase 2] Synthesizing original design...")
            
            start = time.time()
            orig_synth = self.synth_tool.run(original_rtl, top_module)
            duration = (time.time() - start) * 1000
            
            self.trajectory.add_step(
                agent="Scorer",
                action="tool_call",
                tool_name="rtl_synthesis",
                input_summary=f"Synthesize original {top_module}",
                output_summary=f"Cells={orig_synth['metrics'].get('area_cells', 'N/A')}",
                success=orig_synth['success'],
                duration_ms=duration
            )
            
            if not orig_synth['success']:
                result.error = f"Original synthesis failed: {orig_synth.get('errors')}"
                return result
            
            result.original_metrics = orig_synth['metrics']
            print(f"  Original: {orig_synth['metrics'].get('area_cells', 'N/A')} cells")
            
            # ===== Phase 3: Generate Optimized Design =====
            print("\n[Phase 3] Generating optimized design...")
            
            start = time.time()
            if self.use_buggy_optimizer:
                optimized_rtl = mock_optimizer_response(original_rtl, analysis)
            else:
                optimized_rtl = mock_optimizer_response_correct(original_rtl, analysis)
            duration = (time.time() - start) * 1000
            
            self.trajectory.add_step(
                agent="Optimizer",
                action="llm_generate",
                tool_name=None,
                input_summary="Generate optimized RTL",
                output_summary=f"Generated {len(optimized_rtl)} chars of RTL",
                success=True,
                duration_ms=duration,
                metadata={"buggy": self.use_buggy_optimizer}
            )
            
            # Lint the optimized design
            start = time.time()
            opt_lint = self.lint_tool.run(optimized_rtl)
            duration = (time.time() - start) * 1000
            
            self.trajectory.add_step(
                agent="Optimizer",
                action="tool_call",
                tool_name="rtl_lint",
                input_summary="Lint optimized design",
                output_summary=f"Success={bool(opt_lint.get('module_name'))}",
                success=bool(opt_lint.get('module_name')),
                duration_ms=duration
            )
            
            if opt_lint.get('errors') and not opt_lint.get('module_name'):
                result.error = f"Optimized design has syntax errors: {opt_lint.get('errors')}"
                return result
            
            # ===== Phase 4: Validate Optimized Design =====
            print("\n[Phase 4] Validating optimized design...")
            
            # Need to adjust testbench for new module name
            adjusted_tb = testbench.replace("alu_baseline", "alu_optimized")
            
            start = time.time()
            sim_result = self.sim_tool.run(optimized_rtl, adjusted_tb, "alu_optimized")
            duration = (time.time() - start) * 1000
            
            self.trajectory.add_step(
                agent="Validator",
                action="tool_call",
                tool_name="simulation",
                input_summary="Run functional simulation",
                output_summary=f"Passed={sim_result.get('tests_passed', 0)}, Failed={sim_result.get('tests_failed', 0)}",
                success=sim_result['success'],
                duration_ms=duration
            )
            
            result.validation_passed = sim_result['success']
            
            if not sim_result['success']:
                print(f"  ✗ Validation FAILED: {sim_result.get('tests_failed', 0)} tests failed")
                print(f"    This is a common LLM failure mode - generating incorrect RTL!")
            else:
                print(f"  ✓ Validation PASSED: {sim_result.get('tests_passed', 0)} tests passed")
            
            # ===== Phase 5: Score Optimized Design =====
            print("\n[Phase 5] Scoring optimized design...")
            
            start = time.time()
            opt_synth = self.synth_tool.run(optimized_rtl, "alu_optimized")
            duration = (time.time() - start) * 1000
            
            self.trajectory.add_step(
                agent="Scorer",
                action="tool_call",
                tool_name="rtl_synthesis",
                input_summary="Synthesize optimized design",
                output_summary=f"Cells={opt_synth['metrics'].get('area_cells', 'N/A')}",
                success=opt_synth['success'],
                duration_ms=duration
            )
            
            if opt_synth['success']:
                result.optimized_metrics = opt_synth['metrics']
                
                # Compute improvement
                orig_cells = result.original_metrics.get('area_cells', 1)
                opt_cells = result.optimized_metrics.get('area_cells', 1)
                reduction = (orig_cells - opt_cells) / orig_cells * 100 if orig_cells > 0 else 0
                
                result.improvement = {
                    "area_reduction_pct": reduction,
                    "original_cells": orig_cells,
                    "optimized_cells": opt_cells
                }
                
                print(f"  Optimized: {opt_cells} cells ({reduction:.1f}% reduction)")
            
            result.success = True
            return result
            
        except Exception as e:
            result.error = str(e)
            return result


# ============================================================
# Entry Point
# ============================================================

def main():
    """Run the demonstration."""
    
    print("="*60)
    print("SoC Optimization Agent Demonstration")
    print("="*60)
    
    # Load the baseline design
    designs_dir = Path(__file__).parent / "designs"
    
    original_rtl = (designs_dir / "alu_baseline.v").read_text()
    testbench = (designs_dir / "alu_tb.v").read_text()
    
    print("\n" + "-"*60)
    print("RUN 1: Using BUGGY optimizer (demonstrates LLM failures)")
    print("-"*60)
    
    pipeline_buggy = OptimizationPipeline(
        work_dir=Path("./run_buggy"),
        use_buggy_optimizer=True
    )
    result_buggy = pipeline_buggy.run(original_rtl, testbench)
    
    print(result_buggy.summary())
    result_buggy.trajectory.print_summary()
    
    print("\n" + "-"*60)
    print("RUN 2: Using CORRECT optimizer (demonstrates success case)")
    print("-"*60)
    
    pipeline_correct = OptimizationPipeline(
        work_dir=Path("./run_correct"),
        use_buggy_optimizer=False
    )
    result_correct = pipeline_correct.run(original_rtl, testbench)
    
    print(result_correct.summary())
    result_correct.trajectory.print_summary()
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS: What This Demonstrates")
    print("="*60)
    
    print("""
The two runs above show the DUAL NATURE of LLM-based optimization:

RUN 1 (Buggy) - COMMON LLM FAILURE MODES:
-----------------------------------------
1. The LLM correctly identified optimization opportunities
2. It correctly implemented add/sub resource sharing
3. BUT it introduced a BUG in the shifter logic!
   - Original: a << b[2:0] (shift by 0-7 positions)
   - Buggy:    a << 1      (always shift by 1)
4. The validation step CAUGHT this bug via simulation
5. Without validation, this bug would have shipped!

RUN 2 (Correct) - WHEN LLMs WORK WELL:
--------------------------------------
1. Same analysis, same optimization opportunities
2. Correctly preserved shifter functionality
3. Still achieved area reduction via add/sub sharing
4. Validation passed

KEY INSIGHTS:
-------------
1. LLMs ARE useful for suggesting optimizations
2. LLMs CANNOT be trusted to implement them correctly
3. Ground-truth validation (simulation) is ESSENTIAL
4. Trajectory tracking shows WHERE things went wrong
5. The agent architecture (analyze→optimize→validate→score)
   provides defense-in-depth against LLM errors
""")
    
    return result_buggy, result_correct


if __name__ == "__main__":
    main()
