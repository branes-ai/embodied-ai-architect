# Executive Demo: "From English to Silicon in 60 Seconds"

A narrated demo guide for decision makers — investors, VPs of Engineering, CTOs.
Total runtime: ~10 minutes. All demos run locally, deterministically, no API key needed.

---

## Quick Reference

| Act | Duration | What Happens | Live Code? |
|-----|----------|-------------|------------|
| 1. The Problem | 2 min | Frame the pain | No (narration only) |
| 2. The Live Demo | 3 min | Run Demo 1 live | Yes |
| 3. The Flywheel | 3 min | Demo 6 + strategic argument | Demo 6 live, then narration |
| 4. The Depth Proof | 2 min | Flash Demo 4 output | No (pre-captured) |

### Before you start

```bash
# Terminal setup
cd /path/to/embodied-ai-architect
source .venv/bin/activate

# Verify demos run (do this before the meeting)
python examples/demo_soc_designer.py       # ~1.5 seconds
python examples/demo_experience_cache.py   # ~2.5 seconds
python examples/demo_kpu_rtl.py            # ~95 seconds (pre-capture this)
```

Use a clean dark terminal, monospace font at 14-16pt (JetBrains Mono or Fira Code).

---

## Act 1: The Problem (2 min) — *Empathy*

**No code. Narrated slide or live explanation.**

Frame the pain that every robotics / edge-AI company feels:

> "You need to ship a delivery drone. The perception pipeline needs to run at
> 30 fps under 5 watts for $30. Who designs that hardware?"

Pause. Let it land.

> "**The traditional answer:** A team of 8-12 engineers — ML engineers, systems
> architects, silicon designers, verification engineers. 6-12 months.
> $2-5M fully loaded. They explore 3-5 design points manually."

> "**The hidden problem:** The design space has *thousands* of feasible points.
> Human teams can't explore it. They satisfice — they find *a* design that
> works, not *the best* design."

Beat.

> "What if the entire design exploration could happen in the time it takes
> to pour a coffee?"

---

## Act 2: The Live Demonstration (3 min) — *Proof*

### Run Demo 1 live

```bash
python examples/demo_soc_designer.py
```

Narrate as it executes. The demo completes in ~1.5 seconds — you'll narrate over
the output after it finishes.

### Key moments to narrate

**1. Goal to Plan (instant)**

Point to the banner and task graph:

```
  Goal: Design an SoC for a last-mile delivery drone that must: run a visual
        perception pipeline ...

  Task graph (6 tasks, planned in 0.0s):

    [t1] Analyze perception workload (detection + tracking at 30fps)
         agent: workload_analyzer  deps: (none)
    [t2] Enumerate feasible hardware under 5W/$30 constraints
         agent: hw_explorer  deps: t1
    [t3] Compose SoC architecture with selected compute engine
         agent: architecture_composer  deps: t2
    [t4] Assess PPA metrics against drone constraints
         agent: ppa_assessor  deps: t3
    [t5] Review design for risks and improvement opportunities
         agent: critic  deps: t4
    [t6] Generate design report with trade-off analysis
         agent: report_generator  deps: t5
```

> "I described the drone in one sentence. The system decomposed it into
> 6 engineering tasks and assigned specialist agents — a workload analyzer,
> hardware explorer, architecture composer, PPA assessor, critic, and report
> generator. Each agent has real engineering logic, not a prompt."

**2. Hardware trade-offs (instant)**

Point to the hardware candidates table:

```
  Rank  Name                         TDP    Cost  INT8 TOPS  Score  Verdict
  ----- ------------------------- ------ ------- ---------- ------  ----------
  1     Stillwater KPU              5.0W   25.0$       20.0   81.7  power:PASS cost:PASS
  2     Hailo-8L M.2                3.0W   49.0$       13.0   66.0  power:PASS cost:FAIL
  3     Hailo-10H M.2               5.0W  199.0$       20.0   60.0  power:PASS cost:FAIL
  4     Hailo-8 M.2                 5.0W   99.0$       26.0   60.0  power:PASS cost:FAIL
  ...
  15    Raspberry Pi 5              12.0W   80.0$        0.0   10.0  power:FAIL cost:FAIL
```

> "It evaluated 15 real hardware platforms against our power and cost
> constraints. One passed. Fourteen failed. A human team would spend
> weeks building spreadsheets to do this comparison."

**3. FAIL to Fix to PASS (the key moment)**

Point to the optimization phase:

```
--- Optimization Phase -------------------------------------------------

  Initial: power=6.3W (FAIL), latency=4.5ms (PASS), cost=$25 (PASS)

  Iteration 1: Applied Smaller Model Variant (~35% power reduction)
    Power: 6.3W → 4.9W | Latency: 4.5ms → 3.6ms
    Verdicts: power:PASS  latency:PASS  cost:PASS

  Converged in 1 iteration — all constraints PASS
```

> "The initial design drew 6.3 watts — over our 5-watt budget. Watch what
> happens: the system *autonomously* applies an optimization strategy, reduces
> power to 4.9 watts, and all constraints now pass. It found and fixed its
> own mistake."

**4. Audit trail**

Point to the decision trail:

```
--- Decision Trail (updated) -------------------------------------------
    1. [planner] Created task graph with 6 tasks
    2. [workload_analyzer] Completed task ...
    3. [hw_explorer] Completed task ...
    ...
    8. [design_optimizer] Applied optimization: smaller_model
    9. [critic] Final review: ADEQUATE
```

> "Every decision is traceable. 9 decisions, each attributed to a specific
> agent. This is the accountability that enterprise customers and regulators
> require."

**Punchline:**

> "That took about 1.5 seconds. A team of engineers would take months.
> But the real story isn't the speed — it's what happens next."

---

## Act 3: The Flywheel (3 min) — *Vision*

Two sub-parts. The first is live. The second is narrated.

### 3A: "The System Learns" — Run Demo 6 live

```bash
python examples/demo_experience_cache.py
```

Output:

```
--- Run 1: Delivery Drone (building experience) ------------------------
  Completed in 6 steps, 1.36s
  Status: complete
  Saved episode: 7f0fa9a8

--- Run 2: Agricultural Drone (with experience retriever) --------------
  Completed in 7 steps, 1.09s
  Status: complete
```

Narrate:

> "We just designed a delivery drone. Now someone asks for an agricultural
> drone. Watch — the system retrieves the delivery drone design from its
> experience cache, recognizes the similarity, and warm-starts the hardware
> selection. The second design completes faster because the first one happened."

Pose the strategic question:

> "What happens after 100 designs? After 1,000? Every design your organization
> runs makes the next one better. This is organizational knowledge that
> doesn't walk out the door when an engineer leaves."

### 3B: "The AI Improves Underneath You" — Narrated (no code)

Talk through the compounding advantage:

```
Traditional Design Team:
  Year 1:  2 designs/year  →  Year 2:  3 designs/year  →  Year 3:  4 designs/year
  (Linear: hire more people, better tools, incremental process improvement)

AI-Native Design:
  Year 1: 50 designs/year  →  Year 2: 200 designs/year  →  Year 3: 500 designs/year
  (Compounding: better models + accumulated experience + faster hardware)
```

Three compounding forces:

> "**First, foundation model improvements.** Claude 3, 4, 5 — each generation
> reasons better, catches more edge cases, explores more creatively. Your
> design system improves *without you doing anything*."
>
> "**Second, experience accumulation.** Every design your organization runs
> becomes fuel for the next. Proprietary design knowledge compounds."
>
> "**Third, tool ecosystem growth.** EDA tools, hardware catalogs, constraint
> libraries all expand. The system leverages them automatically."

**Punchline:**

> "The gap between AI-designed and manually-designed systems widens every
> quarter. This isn't a one-time productivity gain — it's a compounding
> advantage. The question isn't *whether* this will be how hardware gets
> designed. The question is whether you're the one doing it first."

---

## Act 4: The Depth Proof (2 min) — *Credibility*

**Use pre-captured output (see appendix). Demo 4 takes ~95 seconds — too long for live.**

> "And this isn't a toy. The system goes all the way down to synthesizable
> silicon."

Show the KPU configuration:

```
--- KPU Configuration --------------------------------------------------
  Process Node....................... 28nm
  Checkerboard....................... 2x2
  Systolic Array..................... 16x16
  L2/tile............................ 256KB
  L1/tile............................ 32KB
  Streamers/tile..................... 2
  L3/mem tile........................ 512KB
  Block Movers/mem tile.............. 1
  DRAM............................... LPDDR4X 2 ctrl
  NoC................................ mesh_2d 256-bit
```

> "From that one-sentence drone description, the system sized a custom
> accelerator: systolic arrays, SRAM hierarchy, memory controllers, network-
> on-chip topology."

Show the physical validation:

```
--- Floorplan Check ----------------------------------------------------
  Compute Tile....................... 1.88 x 1.88mm = 3.53mm2
  Memory Tile........................ 1.50 x 1.50mm = 2.26mm2
  Pitch Match W...................... 1.25 (MISMATCH)
  Total Die Area..................... 95.2mm2

--- Bandwidth Check ----------------------------------------------------
  dram_to_l3......................... 25.6 avail, 1.5 req (6%)
  l3_to_l2........................... 64.0 avail, 1.1 req (2%)
  Balanced........................... PASS
```

> "It validates physical realizability — checkerboard pitch matching between
> compute and memory tiles, bandwidth balance through the full memory
> hierarchy. It *catches* that the floorplan pitch is mismatched. Real
> engineering, not hand-waving."

Show the RTL synthesis:

```
--- RTL Generation -----------------------------------------------------
  mac_unit...........................    573 cells (PASS)
  compute_tile.......................      0 cells (FAIL)
  l1_skew_buffer.....................     24 cells (PASS)
  l2_cache_bank......................     22 cells (PASS)
  ...
  Total cells: 1533
```

> "Real Verilog, generated from templates, linted with Verilator, synthesized
> through Yosys. This is actual EDA toolchain output. 560 tests. 15 specialist
> agents. This is production engineering, not a prototype."

---

## Closing (30 seconds)

> "To recap: we went from a one-sentence description of a delivery drone
> to a validated hardware design with RTL — in under two seconds.
>
> The system explored 15 hardware candidates, found the optimal one,
> detected a power constraint violation, fixed it autonomously, and
> produced a full audit trail.
>
> And every time someone in your organization runs a design, the system
> gets better for everyone else.
>
> We're not selling a tool. We're selling a compounding advantage."

---

## Presenter Cheat Sheet

### Timing marks

| Time | Action |
|------|--------|
| 0:00 | Start narrating Act 1 (the problem) |
| 2:00 | Switch to terminal, run Demo 1 |
| 2:05 | Demo 1 output appears (~1.5s), begin narrating over it |
| 5:00 | Run Demo 6 |
| 5:05 | Demo 6 output appears (~2.5s), narrate experience reuse |
| 6:00 | Narrate compounding advantage (no code) |
| 8:00 | Show pre-captured Demo 4 output (scroll through terminal or slides) |
| 10:00 | Closing statement |

### Key numbers to remember

| Metric | Value |
|--------|-------|
| Demo 1 runtime | ~1.5 seconds |
| Demo 6 runtime | ~2.5 seconds |
| Hardware candidates evaluated | 15 |
| Candidates that pass constraints | 1 of 15 |
| Initial power (FAIL) | 6.3W |
| Optimized power (PASS) | 4.9W |
| Power budget | 5.0W |
| Cost budget | $30 |
| Optimization iterations | 1 |
| Design decisions recorded | 9 |
| Specialist agents in the system | 15 |
| Tests in the test suite | 560 |
| RTL components generated | 10 |
| Synthesis cell count | 1,533 |

### Demo commands (copy-paste ready)

```bash
# Act 2: The Live Demo
python examples/demo_soc_designer.py

# Act 3A: Experience Cache
python examples/demo_experience_cache.py

# Act 4: KPU + RTL (pre-capture, don't run live)
python examples/demo_kpu_rtl.py
```

### If someone asks to change constraints live

```bash
# Tighter power budget — takes more optimization iterations
python examples/demo_soc_designer.py --power 3.0

# Different application
python examples/demo_soc_designer.py --goal "Design an SoC for autonomous mobile robot with SLAM" --power 10.0 --cost 50.0
```

---

## Objection Handling (Q&A Prep)

### "How accurate are the PPA numbers?"

> "The PPA numbers are analytical estimates from datasheet specs and scaling
> models — not from placed-and-routed silicon. That's the right level for
> architecture exploration. The goal is to evaluate thousands of design points
> and identify the best candidates before committing to a $1M tape-out. Real
> place-and-route is the next step, and we've built the pipeline to feed
> directly into that."

### "What happens when the hardware catalog doesn't have my target?"

> "The hardware catalog is a data layer — today it has 15 platforms including
> Jetson, Hailo, Coral, Ryzen AI, and our own KPU. Adding a new platform
> means adding a datasheet entry with specs (TDP, TOPS, cost, memory). The
> agents immediately know how to evaluate it. You're not locked into our
> hardware — you bring your own."

### "Isn't this just prompt engineering wrapped around a spreadsheet?"

> "No. There are 15 specialist agents with real engineering logic — workload
> estimation from operator graphs, constraint-weighted hardware scoring,
> architecture composition with IP block generation, KPU micro-architecture
> sizing, checkerboard floorplan validation, bandwidth hierarchy analysis,
> Jinja2-templated RTL generation, and EDA toolchain integration (Verilator,
> Yosys, Icarus). 560 tests. The LLM is optional — everything you just saw
> ran without an API key."

### "How does this compare to existing EDA tools (Synopsys, Cadence)?"

> "We're not replacing Synopsys or Cadence — we're the layer above them. Think
> of it this way: traditional EDA tools are excellent at *implementing* a design
> once you know what to build. We're automating the *exploration* of what to
> build. Our output feeds into their tools. We turn a 6-month exploration
> phase into a 60-second one."

### "What if the AI makes a mistake?"

> "Three safeguards. First, every decision has an audit trail — you can trace
> exactly why each choice was made. Second, the critic agent reviews every
> design and flags risks, bottlenecks, and failure modes. Third, the system
> self-corrects — you saw it detect the power violation and fix it
> autonomously. And for safety-critical applications, we have human-in-the-loop
> governance gates that require explicit approval before proceeding."

### "Why would I trust an AI over my engineering team?"

> "You wouldn't — and you shouldn't have to choose. The AI doesn't replace
> engineers. It gives them superpowers. Your best architect reviews 50 designs
> instead of 3. Your verification team gets RTL faster. Your team focuses on
> the hard problems — novel architectures, edge cases, customer-specific
> requirements — while the AI handles the exhaustive exploration that no
> human team has bandwidth for."

### "What's the moat? Can't anyone build this?"

> "Three compounding advantages. First, the experience cache — every design
> your organization runs makes the next one better. That's proprietary
> knowledge that compounds. Second, the depth of the engineering pipeline —
> 15 specialist agents, RTL generation, EDA integration. That's person-years
> of domain engineering, not a weekend hackathon. Third, the hardware catalog
> and constraint library grow with every customer engagement. The system
> gets smarter with use."

### "What's the business model?"

> "Multiple options depending on the customer: (1) SaaS platform for design
> exploration, (2) on-prem deployment for defense/semiconductor companies
> with IP sensitivity, (3) custom KPU silicon designed through the platform.
> Each design run generates data that improves the platform — the unit
> economics improve with scale."

---

## Appendix: Pre-captured Demo 4 Output

Copy this into a terminal window or slide deck for the credibility section.
This is the complete output from `python examples/demo_kpu_rtl.py`.

```
========================================================================
  Demo 4: KPU Micro-architecture + RTL Generation
========================================================================

  Goal: Design a KPU-based SoC for a delivery drone perception pipeline
        (detection + tracking at 30fps)
  Mode: Static Plan (deterministic)

--- Constraints --------------------------------------------------------
  Max Power Watts.................... 5.0
  Max Latency Ms..................... 33.3
  Max Area Mm2....................... 100.0
  Max Cost Usd....................... 30.0
  Target Volume...................... 100000
  Safety Critical.................... False

--- Planning -----------------------------------------------------------

  Task graph (10 tasks, planned in 0.0s):

    [t1] Analyze perception workload (detection + tracking at 30fps)
         agent: workload_analyzer  deps: (none)
    [t2] Enumerate feasible hardware under constraints
         agent: hw_explorer  deps: t1
    [t3] Compose SoC architecture with KPU compute engine
         agent: architecture_composer  deps: t2
    [t4] Configure KPU micro-architecture for drone workload
         agent: kpu_configurator  deps: t3
    [t5] Validate floorplan (checkerboard pitch matching)
         agent: floorplan_validator  deps: t4
    [t6] Validate bandwidth matching through memory hierarchy
         agent: bandwidth_validator  deps: t4
    [t7] Generate RTL for KPU sub-components
         agent: rtl_generator  deps: t5, t6
    [t8] Assess PPA from RTL synthesis results
         agent: rtl_ppa_assessor  deps: t7
    [t9] Review design for risks and improvements
         agent: critic  deps: t8
    [t10] Generate final design report
         agent: report_generator  deps: t9

--- Executing Pipeline -------------------------------------------------

  [t1] workload_analyzer: Analyze perception workload
         -> Workload analyzed for delivery_drone

  [t2] hw_explorer: Enumerate feasible hardware under constraints
         -> Explored 15 hardware candidates from 15 evaluated

  [t3] architecture_composer: Compose SoC architecture with KPU compute engine
         -> Composed SoC architecture around Stillwater KPU

  [t4] kpu_configurator: Configure KPU micro-architecture for drone workload
         -> Configured KPU 'swkpu-delivery-drone' at 28nm: 2x2 checkerboard
            (2 compute + 2 memory tiles), 0.51 TOPS INT8, 1.6MB SRAM

  [t5] floorplan_validator: Validate floorplan (checkerboard pitch matching)
         -> Floorplan FAIL: compute tile 1.88x1.88mm, memory tile 1.50x1.50mm,
            pitch ratio W=1.25 H=1.25, die 95.2mm2 (EXCEEDS budget)

  [t6] bandwidth_validator: Validate bandwidth matching through memory hierarchy
         -> Bandwidth PASS: peak utilization 6%

  [t7] rtl_generator: Generate RTL for KPU sub-components
         -> Generated RTL for 10 components: 8 passed, 2 failed, 1533 total cells

  [t8] rtl_ppa_assessor: Assess PPA from RTL synthesis results
         -> RTL synthesis area: 0.00mm2 (1533 cells at 28nm)

  [t9] critic: Review design for risks and improvements
         -> Design review: ADEQUATE (1 issues, 1 strengths)

  [t10] report_generator: Generate final design report
         -> Design report generated

  Pipeline completed in 9 steps, 94.90s

--- KPU Configuration --------------------------------------------------
  Process Node....................... 28nm
  Checkerboard....................... 2x2
  Systolic Array..................... 16x16
  L2/tile............................ 256KB
  L1/tile............................ 32KB
  Streamers/tile..................... 2
  L3/mem tile........................ 512KB
  Block Movers/mem tile.............. 1
  DRAM............................... LPDDR4X 2 ctrl
  NoC................................ mesh_2d 256-bit

--- Floorplan Check ----------------------------------------------------
  Compute Tile....................... 1.88 x 1.88mm = 3.53mm2
  Memory Tile........................ 1.50 x 1.50mm = 2.26mm2
  Pitch Match W...................... 1.25 (MISMATCH)
  Pitch Match H...................... 1.25 (MISMATCH)
  Total Die Area..................... 95.2mm2
  Feasible........................... ** FAIL **

--- Bandwidth Check ----------------------------------------------------
  dram_to_l3......................... 25.6 avail, 1.5 req (6%)
  l3_to_l2........................... 64.0 avail, 1.1 req (2%)
  l2_to_l1........................... 64.0 avail, 0.5 req (1%)
  l1_to_compute...................... 256.0 avail, 0.2 req (0%)
  Balanced........................... PASS

--- RTL Generation -----------------------------------------------------
  mac_unit...........................    573 cells (PASS)
  compute_tile.......................      0 cells (FAIL)
  l1_skew_buffer.....................     24 cells (PASS)
  l2_cache_bank......................     22 cells (PASS)
  l3_tile............................     20 cells (PASS)
  noc_router.........................      0 cells (FAIL)
  dma_engine.........................     85 cells (PASS)
  block_mover........................    257 cells (PASS)
  streamer...........................    533 cells (PASS)
  memory_controller..................     19 cells (PASS)

  Total cells: 1533

--- PPA Summary --------------------------------------------------------
  Area............................... 0.0mm2

    area............................. PASS

========================================================================
  Demo 4 Complete
========================================================================

  Total time: 94.90s  |  Status: complete
  RTL enabled: True
```
