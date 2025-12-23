Great question. Let me build a complete, runnable example that exposes both the power and the pitfalls of agentic workflows.

## The Example: Optimizing a Simple ALU

I'll use a small Arithmetic Logic Unit because:
- It's small enough to fit in LLM context
- It has real optimization opportunities (carry chains, resource sharing)
- It can be evaluated with actual open-source EDA tools
- It will expose where LLMs hallucinate vs. where they genuinely help

Let me create the complete working system:I see the pipeline failed early because of issues in the lint step. Let me fix the pipeline to be more robust and show the full demo:The demo is working. Let me make the simulation more realistic by having it actually detect the shifter bug:Let me fix the pattern matching to properly detect the bug:

---
Let me package everything into a single downloadable archive for you:Here are both formats:

**To use:**

```bash
# Extract
unzip soc_optimization_demo.zip
cd soc_optimization

# Install dependency
pip install pydantic

# Run
python demo.py
```

**Contents:**
- `designs/alu_baseline.v` — Original ALU RTL
- `designs/alu_tb.v` — Testbench
- `tools.py` — EDA tool wrappers
- `pipeline.py` — Agent workflow + mock LLM responses
- `trajectory_eval.py` — Trajectory evaluation
- `demo.py` — Main demonstration

The demo uses mock EDA tools by default. If you have Yosys and Icarus Verilog installed, it will automatically use real synthesis and simulation.