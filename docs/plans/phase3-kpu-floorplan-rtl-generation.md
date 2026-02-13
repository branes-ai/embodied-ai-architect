# Phase 3: KPU Micro-architecture Configuration, Floorplan Validation & RTL Generation

## Context

Phase 2 built the architecture-level optimization loop (hardware selection, system-level PPA). But the system picks "KPU" as a black box — it doesn't configure the KPU's internal micro-architecture, doesn't verify the design fits on a die, and doesn't check that data flow is balanced. Without these, RTL synthesis is premature.

Phase 3 adds three layers **before** RTL:

1. **KPU Micro-architecture Configuration** — size all constituent parts: DRAM technology, # memory controllers, DMA engines, L3 tile size, NoC, Block Movers, L2 size/banks, Streamer, L1 skew buffer, Compute Tile dimensions and arithmetic
2. **Floorplan Validation** — 2D pitch assessment: do all components fit at the target process node?
3. **Bandwidth Matching** — ingress/egress validation: is data flow balanced from DRAM through L3→L2→L1→compute and back?

Only after a valid KPU configuration passes floorplan and bandwidth checks do we enter RTL synthesis.

**Four-level architecture after Phase 3:**
```
Outer Loop (soc_graph.py):       planner → dispatch → evaluate → [optimize|report]
  Middle Loop (dispatcher.py):   DAG-aware task execution (specialist agents)
    KPU Loop (kpu_loop.py):      configure → floorplan check → bandwidth check → [adjust|accept]
      RTL Loop (rtl_loop.py):    select component → lint → synthesize → validate → critique
```

All new features are **opt-in** via `rtl_enabled=True`. Existing demos and tests work unchanged.

## Existing Infrastructure to Reuse

| Component | Location | What it provides |
|-----------|----------|-----------------|
| KPU spec | `agents/deployment/targets/kpu/spec.py` | `KPUConfig`, `MemoryConfig`, `ComputeConfig`, precision enums |
| Area model | `../graphs/src/graphs/hardware/soc_infrastructure.py` | `compute_alu_area()`, `compute_sram_area()`, SRAM density tables, interconnect scaling |
| Roofline | `../graphs/src/graphs/estimation/roofline.py` | Bandwidth efficiency, compute/memory bottleneck analysis |
| Data movement | `../graphs/src/graphs/research/dataflow/data_movement.py` | Per-level energy, dataflow analysis (weight/output/row stationary) |
| SRAM density | `../graphs/docs/reference/chip_area_estimates.md` | SRAM Mb/mm² by process node (2nm–180nm) |
| HW schemas | `../embodied-schemas/src/embodied_schemas/hardware.py` | `HardwareCapability` with bandwidth, cache hierarchy |
| Technology DB | `experiments/langgraph/soc_optimizer/workflow.py` | `TECHNOLOGY_NODES` (2nm–180nm), reference cell areas |
| EDA tools | `experiments/langgraph/soc_optimizer/tools/` | Verilator lint, Yosys synthesis, Icarus simulation (with mock fallbacks) |

## Implementation Steps (16 steps)

---

### Phase 3a: KPU Micro-architecture (Steps 1–8)

---

### Step 1: Technology Node Database

**Create** `src/embodied_ai_architect/graphs/technology.py`

Port `TECHNOLOGY_NODES` from `experiments/langgraph/soc_optimizer/workflow.py` (lines 47–360) plus SRAM density data from `../graphs/docs/reference/chip_area_estimates.md`.

Contents:
- `TECHNOLOGY_NODES: dict` — full 2nm–180nm process database (cell area, gate delay, energy, Vdd, reference areas)
- `SRAM_DENSITY: dict[int, float]` — Mb/mm² by process node (from chip_area_estimates.md: 16nm=18, 12nm=32, 7nm=45, 5nm=48, 2nm=38.1)
- `TechnologyNode` Pydantic model — typed wrapper
- `get_technology(process_nm) -> TechnologyNode` — nearest-match lookup
- `estimate_area_um2(cell_count, process_nm)`, `estimate_sram_area_mm2(size_bytes, process_nm)`, `estimate_timing_ps(logic_levels, process_nm)`

**Create** `tests/test_technology.py` — ~6 tests

---

### Step 2: KPU Micro-architecture Configuration Model

**Create** `src/embodied_ai_architect/graphs/kpu_config.py`

A detailed micro-architecture configuration extending the existing high-level `KPUConfig` concept. This is the **configuration space** that the KPU configurator explores.

```python
class DRAMConfig(BaseModel):
    """External memory configuration."""
    technology: str = "LPDDR4X"           # LPDDR4X, LPDDR5, HBM2E
    num_controllers: int = 2              # Number of memory controllers
    channels_per_controller: int = 2      # Channels per controller
    bandwidth_per_channel_gbps: float = 6.4  # Per-channel bandwidth
    capacity_gb: float = 4.0

    @property
    def total_bandwidth_gbps(self) -> float:
        return self.num_controllers * self.channels_per_controller * self.bandwidth_per_channel_gbps

class DMAConfig(BaseModel):
    """DMA engine configuration."""
    num_engines: int = 4
    max_transfer_bytes: int = 1024 * 1024  # 1MB max transfer
    queue_depth: int = 8

class L3Config(BaseModel):
    """Shared L3 cache / global buffer."""
    total_size_bytes: int = 4 * 1024 * 1024    # 4MB
    tile_size_bytes: int = 512 * 1024           # 512KB per tile
    num_banks: int = 8
    bandwidth_gbps: float = 128.0               # Internal bandwidth

class NoCConfig(BaseModel):
    """Network on Chip configuration."""
    topology: str = "mesh_2d"            # mesh_2d, ring, tree, crossbar
    link_width_bits: int = 256           # Data width per link
    frequency_mhz: float = 1000.0        # NoC clock
    num_routers: int = 16

    @property
    def link_bandwidth_gbps(self) -> float:
        return self.link_width_bits * self.frequency_mhz / 8 / 1000

class BlockMoverConfig(BaseModel):
    """Block mover for bulk data transfers between memory levels."""
    num_movers: int = 4
    transfer_bandwidth_gbps: float = 32.0  # Per mover

class ComputeTileConfig(BaseModel):
    """Compute tile configuration.

    A compute tile is a self-contained unit containing:
    - Systolic array (MAC array for matrix ops)
    - Vector unit (element-wise ops)
    - L2 cache banks (inside the tile to minimize wire loading by streamers)
    - L1 skew buffer (scratchpad closest to compute)
    - Streamers (prefetch engines feeding L1 from L2)
    - Local control/scheduler

    L2 is INSIDE the compute tile (not separate) to minimize capacitive
    loading on the streamer-to-L2 path.
    """
    num_tiles: int = 4                     # Number of compute tiles in the array
    # --- Systolic array ---
    array_rows: int = 16                   # Systolic array rows
    array_cols: int = 16                   # Systolic array cols
    vector_lanes: int = 16                 # Vector unit width
    frequency_mhz: float = 500.0
    supported_precisions: list[str] = ["int8", "fp16", "bf16"]
    # --- L2 (inside compute tile) ---
    l2_size_bytes: int = 256 * 1024        # 256KB L2 per tile
    l2_num_banks: int = 8
    l2_read_ports: int = 2
    l2_write_ports: int = 1
    # --- L1 skew buffer (inside compute tile) ---
    l1_size_bytes: int = 32 * 1024         # 32KB L1 per tile
    l1_num_banks: int = 4
    # --- Streamers (inside compute tile) ---
    num_streamers: int = 2
    streamer_prefetch_depth: int = 4
    streamer_buffer_bytes: int = 16 * 1024 # 16KB prefetch buffer

    @property
    def peak_tops_int8(self) -> float:
        ops_per_cycle = self.array_rows * self.array_cols * 2  # multiply + accumulate
        total_ops = ops_per_cycle * self.num_tiles * self.frequency_mhz * 1e6
        return total_ops / 1e12

class MemoryTileConfig(BaseModel):
    """Memory tile (L3) configuration.

    Memory tiles alternate with compute tiles in a checkerboard pattern.
    Each memory tile contains:
    - L3 SRAM banks (shared cache / global buffer)
    - Block movers (bulk data transfer between L3 and DRAM/L2)
    - DMA engines (for DRAM↔L3 transfers)

    The memory tile width/height must pitch-match the compute tile
    for efficient checkerboard layout.
    """
    l3_tile_size_bytes: int = 512 * 1024   # 512KB of L3 SRAM per memory tile
    l3_num_banks: int = 4                  # Banks per memory tile
    num_block_movers: int = 2              # Block movers per memory tile
    block_mover_bw_gbps: float = 32.0      # Per mover bandwidth
    num_dma_engines: int = 1               # DMA engines per memory tile
    dma_max_transfer_bytes: int = 1024 * 1024
    dma_queue_depth: int = 8

class KPUMicroArchConfig(BaseModel):
    """Complete KPU micro-architecture configuration.

    The KPU is organized as a 2D checkerboard of alternating compute tiles
    and memory tiles with nearest-neighbor communication:

        C  M  C  M
        M  C  M  C
        C  M  C  M

    Key design principle: compute tiles and memory tiles must be
    pitch-matched (similar width × height) for efficient 2D layout.
    L2 caches live INSIDE compute tiles to minimize streamer wire loading.
    """
    name: str = "swkpu-v1"
    process_nm: int = 28
    # --- External memory ---
    dram: DRAMConfig = Field(default_factory=DRAMConfig)
    # --- Network on Chip (between tiles) ---
    noc: NoCConfig = Field(default_factory=NoCConfig)
    # --- Compute tiles (contain L2 + L1 + streamers + array) ---
    compute_tile: ComputeTileConfig = Field(default_factory=ComputeTileConfig)
    # --- Memory tiles (contain L3 + block movers + DMA) ---
    memory_tile: MemoryTileConfig = Field(default_factory=MemoryTileConfig)
    # --- Array dimensions ---
    array_rows: int = 3                    # Rows in the checkerboard
    array_cols: int = 3                    # Cols in the checkerboard

    @property
    def num_compute_tiles(self) -> int:
        """Compute tiles in checkerboard (ceil of half the grid)."""
        total = self.array_rows * self.array_cols
        return (total + 1) // 2

    @property
    def num_memory_tiles(self) -> int:
        """Memory tiles in checkerboard (floor of half the grid)."""
        total = self.array_rows * self.array_cols
        return total // 2

    @property
    def total_l3_bytes(self) -> int:
        return self.num_memory_tiles * self.memory_tile.l3_tile_size_bytes

    @property
    def total_l2_bytes(self) -> int:
        return self.num_compute_tiles * self.compute_tile.l2_size_bytes

    @property
    def total_l1_bytes(self) -> int:
        return self.num_compute_tiles * self.compute_tile.l1_size_bytes

    @property
    def total_sram_bytes(self) -> int:
        return self.total_l1_bytes + self.total_l2_bytes + self.total_l3_bytes

    @property
    def total_dram_bandwidth_gbps(self) -> float:
        return self.dram.total_bandwidth_gbps
```

Also provide:
- `KPU_PRESETS: dict[str, KPUMicroArchConfig]` — named configurations: "drone_minimal" (2 tiles, 2MB L3, LPDDR4X), "edge_balanced" (4 tiles, 4MB L3), "server_max" (16 tiles, 16MB L3, HBM2E)
- `create_kpu_config(use_case, constraints, workload) -> KPUMicroArchConfig` — heuristic sizing based on workload GFLOPS and constraint budget

**Create** `tests/test_kpu_config.py` — ~8 tests (presets, properties, custom configs, heuristic sizing)

---

### Step 3: Floorplan Estimator (Checkerboard Pitch Matching)

**Create** `src/embodied_ai_architect/graphs/floorplan.py`

The KPU uses a **checkerboard layout** of alternating compute tiles and memory tiles. The floorplan estimator's primary job is to **dimension each tile type** (width × height) and check that they **pitch-match** — i.e., compute tile dimensions ≈ memory tile dimensions so they tile efficiently in 2D.

```
Checkerboard layout:       Pitch matching requirement:
  C  M  C  M               ┌──────┬──────┐
  M  C  M  C               │Comp. │Mem.  │  width_C ≈ width_M
  C  M  C  M               │Tile  │Tile  │  height_C ≈ height_M
                            └──────┴──────┘
```

**Estimation tables** for sub-component areas at each process node (referencing `soc_infrastructure.py` and `chip_area_estimates.md`):

```python
# Area estimation tables (mm² at reference process, scales with node²)
COMPONENT_AREA_TABLES = {
    # --- Inside Compute Tile ---
    "systolic_array": lambda rows, cols, nm: rows * cols * ALU_AREA_5NM * (nm/5)**2 * SYSTOLIC_MULT,
    "vector_unit": lambda lanes, nm: lanes * ALU_AREA_5NM * (nm/5)**2 * 2.0,
    "l2_sram": lambda bytes, nm: estimate_sram_area_mm2(bytes, nm),
    "l1_sram": lambda bytes, nm: estimate_sram_area_mm2(bytes, nm),
    "streamers": lambda count, nm: count * 0.01 * (nm/5)**2,  # Small control logic
    "tile_control": lambda nm: 0.05 * (nm/5)**2,  # Scheduler, pipeline control
    # --- Inside Memory Tile ---
    "l3_sram": lambda bytes, nm: estimate_sram_area_mm2(bytes, nm),
    "block_movers": lambda count, nm: count * 0.02 * (nm/5)**2,
    "dma_engines": lambda count, nm: count * 0.03 * (nm/5)**2,
    "mem_tile_control": lambda nm: 0.02 * (nm/5)**2,
}

class TileDimensions(BaseModel):
    """2D dimensions of a single tile."""
    width_mm: float
    height_mm: float
    area_mm2: float            # width × height
    sub_blocks: list[dict]     # [{name, area_mm2, width_mm, height_mm}]

class FloorplanEstimate(BaseModel):
    """Checkerboard floorplan estimate."""
    compute_tile: TileDimensions    # Single compute tile dimensions
    memory_tile: TileDimensions     # Single memory tile dimensions
    pitch_matched: bool             # width/height within tolerance
    pitch_ratio_width: float        # compute_width / memory_width (ideal = 1.0)
    pitch_ratio_height: float       # compute_height / memory_height (ideal = 1.0)
    pitch_tolerance: float          # Acceptable ratio range (e.g., 0.85–1.15)
    # --- Overall die ---
    array_width_mm: float           # array_cols × tile_width
    array_height_mm: float          # array_rows × tile_height
    core_area_mm2: float            # Array area (tiles only)
    periphery_area_mm2: float       # DRAM controllers, NoC routers at edges, I/O
    total_area_mm2: float           # core + periphery + routing overhead
    die_edge_mm: float              # max(array_width, array_height) + periphery margin
    feasible: bool                  # Fits in die budget AND pitch-matched
    max_die_area_mm2: float
    issues: list[str]               # Pitch mismatch details, area violations, suggestions

def estimate_tile_dimensions(
    tile_config: ComputeTileConfig | MemoryTileConfig,
    process_nm: int,
    aspect_ratio: float = 1.0,     # Target width/height ratio
) -> TileDimensions:
    """Build a 2D facsimile of a tile from its sub-components.

    For each sub-block, estimate area using COMPONENT_AREA_TABLES.
    Arrange sub-blocks in a 2D layout (rows of components) to get
    width and height. The aspect ratio can be tuned to match the
    partner tile.
    """

def estimate_floorplan(
    config: KPUMicroArchConfig,
    max_die_area_mm2: float = 100.0,
    pitch_tolerance: float = 0.15,   # 15% mismatch allowed
    routing_overhead: float = 0.20,   # 20% for inter-tile wires
) -> FloorplanEstimate:
    """Estimate checkerboard floorplan with pitch matching.

    Steps:
    1. Estimate compute tile dimensions (systolic array + L2 + L1 + streamers + control)
    2. Estimate memory tile dimensions (L3 SRAM + block movers + DMA + control)
    3. Check pitch matching: |width_ratio - 1.0| < tolerance AND |height_ratio - 1.0| < tolerance
    4. Compute die dimensions: array_cols × max(tile_widths), array_rows × max(tile_heights)
    5. Add periphery (DRAM controllers, I/O pads) around the array edges
    6. Check total area against die budget

    If pitch mismatch: report which dimension is off and by how much,
    with suggestions (e.g., "increase L3 bank count to widen memory tile"
    or "reduce systolic array cols to narrow compute tile").
    """
```

**2D pitch check**: The primary check is NOT `sqrt(total_area)` but rather:
- `pitch_ratio_width = compute_tile.width_mm / memory_tile.width_mm` → must be in `[1-tol, 1+tol]`
- `pitch_ratio_height = compute_tile.height_mm / memory_tile.height_mm` → must be in `[1-tol, 1+tol]`
- Plus overall die edge < reticle limit (~33mm)

**Create** `tests/test_floorplan.py` — ~8 tests (pitch match pass, pitch mismatch detection, area breakdown, process node scaling, compute tile dimensioning, memory tile dimensioning, aspect ratio tuning, periphery estimation)

---

### Step 4: Bandwidth Matcher

**Create** `src/embodied_ai_architect/graphs/bandwidth.py`

Validates that data flow is balanced through the memory hierarchy. Each interface must provide enough bandwidth to feed the next level without starvation.

```python
class BandwidthLink(BaseModel):
    """One link in the bandwidth chain."""
    name: str                    # "dram_to_l3", "l3_to_l2", "l2_to_l1", "l1_to_compute"
    source: str
    sink: str
    available_gbps: float        # Supply bandwidth
    required_gbps: float         # Demand bandwidth (from workload)
    utilization: float           # required / available
    bottleneck: bool             # utilization > threshold (e.g., 0.85)

class BandwidthMatchResult(BaseModel):
    """Result of bandwidth matching analysis."""
    links: list[BandwidthLink]
    balanced: bool               # No bottlenecks found
    bottleneck_link: Optional[str]  # Name of worst link
    peak_utilization: float
    ingress_gbps: float          # DRAM → chip
    egress_gbps: float           # Chip → DRAM
    compute_demand_gbps: float   # What compute tiles need
    issues: list[str]

def check_bandwidth_match(
    config: KPUMicroArchConfig,
    workload: dict,
    arithmetic_intensity: float = 0.0,  # FLOPs/byte from roofline
) -> BandwidthMatchResult:
    """Check ingress/egress bandwidth matching.

    Chain:
    1. DRAM → memory controllers → L3 (dram.total_bandwidth_gbps)
    2. L3 → block movers/NoC → L2 tiles (l3.bandwidth_gbps, noc.link_bandwidth_gbps)
    3. L2 → streamers → L1 (l2 bandwidth = ports × bank_bandwidth)
    4. L1 → compute (l1 bandwidth vs compute demand)

    Demand side:
    - compute_demand_gbps = peak_tops × bytes_per_op / arithmetic_intensity
    - Derived from workload GFLOPS and data types

    Balance check:
    - Each link utilization < 85% → balanced
    - Any link > 85% → bottleneck identified with recommendation
    """
```

**Create** `tests/test_bandwidth.py` — ~6 tests (balanced config, DRAM bottleneck, NoC bottleneck, workload scaling)

---

### Step 5: KPU Configurator Specialist Agent

**Create** `src/embodied_ai_architect/graphs/kpu_specialists.py`

New specialist agents for the KPU micro-architecture layer:

**`kpu_configurator(task, state) -> dict`**:
- Reads workload profile (GFLOPS, memory requirements, data types) and constraints (power, area, cost)
- Uses `create_kpu_config()` heuristic to generate initial configuration
- Applies workload-specific tuning: vision workloads get larger L2 for feature maps, memory-heavy workloads get more DRAM controllers
- Writes `kpu_config` to state

**`floorplan_validator(task, state) -> dict`**:
- Reads `kpu_config` from state
- Calls `estimate_floorplan()` with target die area from constraints
- Writes `floorplan_estimate` to state
- Reports feasibility verdict (PASS/FAIL) with area breakdown

**`bandwidth_validator(task, state) -> dict`**:
- Reads `kpu_config` and `workload_profile` from state
- Calls `check_bandwidth_match()`
- Writes `bandwidth_match` to state
- Reports balance verdict with bottleneck identification

**`kpu_optimizer(task, state) -> dict`**:
- If **pitch mismatch** (compute tile too wide vs memory tile):
  - Shrink compute: reduce array_cols, reduce L2 banks, or reduce L1 size
  - Grow memory: increase L3 banks per tile, add block movers
  - Or adjust aspect ratio: rearrange sub-blocks within tile
- If **pitch mismatch** (memory tile too tall vs compute tile):
  - Grow compute: increase vector_lanes, add streamer buffer
  - Shrink memory: reduce L3 tile size, fewer DMA engines
- If **die area FAIL**: reduce array_rows/cols (fewer tiles), reduce SRAM sizes
- If **bandwidth FAIL at DRAM**: upgrade DRAM (LPDDR4X → LPDDR5), add controllers
- If **bandwidth FAIL at NoC**: widen links, change topology
- If **bandwidth FAIL at L2/L1**: increase bank count, add read ports
- Records adjustments in working memory (prevents loops)

**Create** `tests/test_kpu_specialists.py` — ~8 tests

---

### Step 6: KPU Validation Loop

**Create** `src/embodied_ai_architect/graphs/kpu_loop.py`

A Python loop (like `rtl_loop.py`) that iterates on KPU configuration until both floorplan and bandwidth pass.

```python
@dataclass
class KPULoopConfig:
    max_iterations: int = 10
    max_die_area_mm2: float = 100.0
    bandwidth_threshold: float = 0.85    # Max utilization before bottleneck

@dataclass
class KPULoopResult:
    success: bool
    config: KPUMicroArchConfig
    floorplan: FloorplanEstimate
    bandwidth: BandwidthMatchResult
    iterations_used: int
    history: list[dict]               # Config snapshots per iteration

def run_kpu_loop(
    workload: dict,
    constraints: dict,
    use_case: str,
    loop_config: KPULoopConfig = None,
) -> KPULoopResult:
    """Configure → floorplan check → bandwidth check → [adjust|accept].

    1. Generate initial KPU config from workload/constraints
    2. Check floorplan (2D pitch)
    3. Check bandwidth matching
    4. If both PASS → accept
    5. If FAIL → apply optimizer adjustments → loop
    """
```

**Create** `tests/test_kpu_loop.py` — ~5 tests (converges, floorplan fix, bandwidth fix, iteration limit)

---

### Step 7: Extend SoCDesignState for KPU + RTL Fields

**Modify** `src/embodied_ai_architect/graphs/soc_state.py`

Add after the existing RTL field (line 210):

```python
# === KPU Micro-architecture ===
kpu_config: dict               # KPUMicroArchConfig serialized
floorplan_estimate: dict       # FloorplanEstimate serialized
bandwidth_match: dict          # BandwidthMatchResult serialized
kpu_optimization_history: list[dict]  # Config snapshots per KPU loop iteration

# === RTL Artifacts ===
rtl_modules: dict[str, str]    # ALREADY EXISTS (line 210)
rtl_testbenches: dict[str, str]
rtl_synthesis_results: dict[str, dict]
rtl_lint_results: dict[str, dict]
rtl_validation_results: dict[str, dict]
rtl_optimization_history: list[dict]
rtl_process_nm: int
rtl_enabled: bool              # Enables KPU config + floorplan + bandwidth + RTL
```

Update `create_initial_soc_state()` with `rtl_enabled=False` default and empty dicts for all new fields.

Add helpers: `get_kpu_config(state)`, `get_floorplan(state)`, `get_bandwidth(state)`, `get_rtl_summary(state)`.

**Extend** `tests/test_soc_state.py` — ~6 tests

---

### Step 8: Register KPU Specialists + Update Planner

**Modify** `src/embodied_ai_architect/graphs/specialists.py`

Register in `create_default_dispatcher()`:
```python
"kpu_configurator": kpu_configurator,
"floorplan_validator": floorplan_validator,
"bandwidth_validator": bandwidth_validator,
"kpu_optimizer": kpu_optimizer,
```

**Modify** `src/embodied_ai_architect/graphs/planner.py`

Add to `PLANNER_SYSTEM_PROMPT` agent table:
```
| kpu_configurator    | Size KPU micro-architecture components for workload                |
| floorplan_validator | Check 2D die area feasibility at target process node               |
| bandwidth_validator | Verify ingress/egress bandwidth matching through memory hierarchy  |
| kpu_optimizer       | Adjust KPU config to fix floorplan/bandwidth violations            |
```

---

### Phase 3b: RTL Generation (Steps 9–16)

---

### Step 9: Port EDA Tool Wrappers

**Create** `src/embodied_ai_architect/graphs/eda_tools/` package

Port from `experiments/langgraph/soc_optimizer/tools/`:
- `eda_tools/__init__.py` — re-exports + `EDAToolchain` facade
- `eda_tools/lint.py` — from `tools/lint.py` (3-tier fallback: Verilator → Yosys → regex)
- `eda_tools/synthesis.py` — from `tools/synthesis.py` (mock fallback). Enhance: add `process_nm` param, use `technology.py` for area scaling
- `eda_tools/simulation.py` — from `tools/simulation.py` (mock fallback)
- `eda_tools/toolchain.py` — NEW facade: `EDAToolchain(work_dir, process_nm)` with `available_tools` property

**Create** `tests/test_eda_tools.py` — ~6 tests (all use mock fallbacks, no EDA tools required)

---

### Step 10: Jinja2 RTL Templates

**Create** `src/embodied_ai_architect/graphs/rtl_templates/` package

Templates for KPU sub-components (matching the `KPUMicroArchConfig` hierarchy):

| KPU Component | Template | Parameters |
|---|---|---|
| Compute Tile | `compute_tile.sv.j2` | array_rows, array_cols, data_width, accum_width |
| MAC Unit | `mac_unit.sv.j2` | data_width, accum_width, pipeline_stages |
| L1 Skew Buffer | `l1_skew_buffer.sv.j2` | size_bytes, num_banks, data_width |
| L2 Cache Bank | `l2_cache_bank.sv.j2` | size_bytes, num_banks, read_ports, write_ports |
| L3 Tile | `l3_tile.sv.j2` | tile_size_bytes, num_banks |
| NoC Router | `noc_router.sv.j2` | num_ports, link_width_bits, buffer_depth |
| DMA Engine | `dma_engine.sv.j2` | max_transfer_bytes, queue_depth |
| Block Mover | `block_mover.sv.j2` | transfer_width_bits |
| Streamer | `streamer.sv.j2` | prefetch_depth, buffer_size_bytes |
| Memory Controller | `memory_controller.sv.j2` | addr_width, data_width, num_channels |
| Register File | `register_file.sv.j2` | num_regs, data_width, read_ports, write_ports |
| ALU | `alu.sv.j2` | data_width, operations |

Files:
- `rtl_templates/__init__.py` — `RTLTemplateEngine` with `render(component_type, params)`, `render_testbench(component_type, name, params)`, `can_generate(component_type)`, `available_templates()`, `get_kpu_components(kpu_config) -> list[dict]` (maps KPU config to template render calls)

Each template has a companion `{name}_tb.sv.j2` testbench.

**Create** `tests/test_rtl_templates.py` — ~6 tests

---

### Step 11: RTL Inner Loop

**Create** `src/embodied_ai_architect/graphs/rtl_loop.py`

Plain Python loop: lint → synthesize → validate → critique per RTL module.

```python
@dataclass
class RTLLoopConfig:
    max_iterations: int = 5
    process_nm: int = 28
    work_dir: Path = None
    skip_validation: bool = False

@dataclass
class RTLLoopResult:
    module_name: str
    success: bool
    rtl_source: str
    testbench_source: Optional[str]
    lint_result: dict
    synthesis_result: dict
    validation_result: Optional[dict]
    metrics: dict   # area_cells, area_um2, num_wires, etc.
    iterations_used: int
    history: list[dict]

def run_rtl_loop(module_name, rtl_source, config, testbench_source=None) -> RTLLoopResult:
```

Phase 3 (template-based, no LLM) runs one iteration per module. Iteration hook is for Phase 4 LLM-based RTL optimization.

**Create** `tests/test_rtl_loop.py` — ~5 tests

---

### Step 12: RTL Specialist Agents

**Create** `src/embodied_ai_architect/graphs/rtl_specialists.py`

**`rtl_generator(task, state) -> dict`**:
- Skips if `rtl_enabled` is False
- Reads `kpu_config` from state (set by `kpu_configurator`)
- Uses `RTLTemplateEngine.get_kpu_components(kpu_config)` to determine which components to generate
- For each component: render Verilog + testbench, run `run_rtl_loop()`
- Writes: `rtl_modules`, `rtl_testbenches`, `rtl_synthesis_results`, `rtl_lint_results`, `rtl_validation_results`

**`rtl_ppa_assessor(task, state) -> dict`**:
- Aggregates synthesis cell counts across all RTL modules
- Uses `technology.py` for area conversion at target process node
- Refines `ppa_metrics.area_mm2` with synthesis-quality numbers
- Cross-checks against floorplan estimate

**Create** `tests/test_rtl_specialists.py` — ~6 tests

---

### Step 13: Update Outer Loop Integration

**Modify** `src/embodied_ai_architect/graphs/soc_graph.py`

Update `_make_report_node` to include KPU config, floorplan, bandwidth, and RTL summary in report when `rtl_enabled`.

No structural changes to outer loop — new specialists dispatched by planner via task graph.

---

### Step 14: Update Experience Cache + Exports

**Modify** `src/embodied_ai_architect/graphs/experience.py`

Add optional fields to `DesignEpisode`:
```python
kpu_config_name: Optional[str] = None
kpu_process_nm: Optional[int] = None
floorplan_area_mm2: Optional[float] = None
bandwidth_balanced: Optional[bool] = None
rtl_modules_generated: int = 0
rtl_total_cells: int = 0
```

**Modify** `src/embodied_ai_architect/graphs/__init__.py` — export all new modules

---

### Step 15: Demo 4

**Create** `examples/demo_kpu_rtl.py`

Demo 4 shows the full four-level flow:

```
Demo 4: KPU Micro-architecture + RTL Generation
=================================================
  Goal: Configure and validate KPU for delivery drone perception

--- KPU Configuration ---
  [kpu_configurator] Sized KPU for 30fps perception:
    Process: 28nm | Checkerboard: 3×3 (5 compute + 4 memory tiles)
    Compute tile: 16×16 systolic + 256KB L2 + 32KB L1 + 2 streamers @ 500MHz
    Memory tile: 512KB L3 + 2 block movers + 1 DMA engine
    Total SRAM: L1 160KB + L2 1.25MB + L3 2MB = 3.4MB
    Peak: 4.1 TOPS INT8
    DRAM: 4GB LPDDR4X, 2 controllers (25.6 GB/s)
    NoC: 2D mesh, 256-bit links @ 1GHz

--- Floorplan Check (Checkerboard Pitch Matching) ---
  [floorplan_validator] Tile dimensioning:
    Compute tile (L2+L1+streamers+16×16 array+control):
      2.1mm × 2.0mm = 4.2 mm²  (L2: 1.8mm², array: 1.6mm², L1: 0.4mm², ctrl: 0.4mm²)
    Memory tile (L3+block movers+DMA+control):
      2.0mm × 1.9mm = 3.8 mm²  (L3: 3.2mm², movers: 0.3mm², DMA: 0.2mm², ctrl: 0.1mm²)
    Pitch match: width 2.1/2.0 = 1.05 ✓  height 2.0/1.9 = 1.05 ✓  (tolerance ±15%)
    Checkerboard: 3×3 grid (5 compute + 4 memory tiles)
    Core: 6.3mm × 6.0mm = 37.8 mm²
    Periphery (DRAM controllers, I/O): 5.6 mm²
    Total: 43.4 mm² (+ 20% routing = 52.1 mm²)
    Die edge: 7.2mm × 7.2mm → PASS (< 100 mm² budget)

--- Bandwidth Check ---
  [bandwidth_validator] Ingress/egress analysis:
    DRAM → L3:     25.6 GB/s available, 18.2 GB/s required (71%) ✓
    L3 → L2 (NoC): 32.0 GB/s available, 22.1 GB/s required (69%) ✓
    L2 → L1:       64.0 GB/s available, 28.0 GB/s required (44%) ✓
    L1 → Compute:  128.0 GB/s available, 32.8 GB/s required (26%) ✓
    Verdict: BALANCED (peak utilization 71%)

--- RTL Generation ---
  [rtl_generator] Generated RTL for 8 KPU components:
    compute_tile     -> 1,240 cells (lint: PASS, synth: PASS)
    mac_unit         ->   234 cells (lint: PASS, synth: PASS)
    l1_skew_buffer   ->   180 cells (lint: PASS, synth: PASS)
    l2_cache_bank    ->   320 cells (lint: PASS, synth: PASS)
    noc_router       ->   450 cells (lint: PASS, synth: PASS)
    ...
  [rtl_ppa_assessor] Synthesis area: 44.2 mm² (floorplan estimate was 46.9 mm²)

--- PPA Summary ---
  Power: 4.8W (PASS) | Area: 44.2 mm² (PASS) | Latency: 28.1ms (PASS)
```

Uses static plan:
```python
DEMO_4_PLAN = [
    {"id": "t1", "agent": "workload_analyzer", ...},
    {"id": "t2", "agent": "hw_explorer", ...},
    {"id": "t3", "agent": "architecture_composer", ...},
    {"id": "t4", "agent": "kpu_configurator", ...},
    {"id": "t5", "agent": "floorplan_validator", "dependencies": ["t4"]},
    {"id": "t6", "agent": "bandwidth_validator", "dependencies": ["t4"]},
    {"id": "t7", "agent": "rtl_generator", "dependencies": ["t5", "t6"]},
    {"id": "t8", "agent": "rtl_ppa_assessor", "dependencies": ["t7"]},
    {"id": "t9", "agent": "critic", "dependencies": ["t8"]},
    {"id": "t10", "agent": "report_generator", "dependencies": ["t9"]},
]
```

---

### Step 16: Integration Tests

**Create** `tests/test_kpu_integration.py` — ~5 tests:
- `test_kpu_loop_converges()` — KPU config passes floorplan + bandwidth
- `test_full_pipeline_with_rtl()` — end-to-end with `rtl_enabled=True`
- `test_backward_compatibility()` — Demo 1/3 unchanged with `rtl_enabled=False`
- `test_floorplan_triggers_resize()` — oversized config gets optimized down
- `test_bandwidth_triggers_upgrade()` — DRAM bottleneck triggers controller addition

**Create** `tests/test_rtl_integration.py` — ~3 tests:
- `test_rtl_from_kpu_config()` — RTL templates driven by KPU config
- `test_experience_episode_includes_kpu()` — KPU fields in saved episode
- `test_optimization_loop_with_kpu_rtl()` — Demo 3 style convergence

All tests use mock EDA fallbacks (no Verilator/Yosys/Icarus required).

---

## Files Summary

| Action | File |
|--------|------|
| **Phase 3a: KPU Micro-architecture** | |
| Create | `src/embodied_ai_architect/graphs/technology.py` |
| Create | `src/embodied_ai_architect/graphs/kpu_config.py` |
| Create | `src/embodied_ai_architect/graphs/floorplan.py` |
| Create | `src/embodied_ai_architect/graphs/bandwidth.py` |
| Create | `src/embodied_ai_architect/graphs/kpu_specialists.py` |
| Create | `src/embodied_ai_architect/graphs/kpu_loop.py` |
| Modify | `src/embodied_ai_architect/graphs/soc_state.py` |
| Modify | `src/embodied_ai_architect/graphs/specialists.py` |
| Modify | `src/embodied_ai_architect/graphs/planner.py` |
| **Phase 3b: RTL Generation** | |
| Create | `src/embodied_ai_architect/graphs/eda_tools/__init__.py` |
| Create | `src/embodied_ai_architect/graphs/eda_tools/lint.py` |
| Create | `src/embodied_ai_architect/graphs/eda_tools/synthesis.py` |
| Create | `src/embodied_ai_architect/graphs/eda_tools/simulation.py` |
| Create | `src/embodied_ai_architect/graphs/eda_tools/toolchain.py` |
| Create | `src/embodied_ai_architect/graphs/rtl_templates/__init__.py` |
| Create | `src/embodied_ai_architect/graphs/rtl_templates/*.sv.j2` (12 templates + 12 testbenches) |
| Create | `src/embodied_ai_architect/graphs/rtl_loop.py` |
| Create | `src/embodied_ai_architect/graphs/rtl_specialists.py` |
| **Integration** | |
| Modify | `src/embodied_ai_architect/graphs/soc_graph.py` |
| Modify | `src/embodied_ai_architect/graphs/experience.py` |
| Modify | `src/embodied_ai_architect/graphs/__init__.py` |
| Create | `examples/demo_kpu_rtl.py` |
| **Tests** | |
| Create | `tests/test_technology.py` |
| Create | `tests/test_kpu_config.py` |
| Create | `tests/test_floorplan.py` |
| Create | `tests/test_bandwidth.py` |
| Create | `tests/test_kpu_specialists.py` |
| Create | `tests/test_kpu_loop.py` |
| Create | `tests/test_eda_tools.py` |
| Create | `tests/test_rtl_templates.py` |
| Create | `tests/test_rtl_loop.py` |
| Create | `tests/test_rtl_specialists.py` |
| Create | `tests/test_kpu_integration.py` |
| Create | `tests/test_rtl_integration.py` |
| Extend | `tests/test_soc_state.py` |

## Scope Boundaries (NOT building in Phase 3)

- No LLM-based RTL generation or KPU configuration — deterministic heuristics + templates only
- No Docker containers for EDA tools — direct subprocess with mock fallback
- No OpenROAD place & route — Yosys synthesis only
- No formal verification (SymbiYosys)
- No Pareto front / design space exploration sweep — single-point configuration
- No liberty/PDK file handling — generic Yosys synthesis
- No timing closure loop (WNS/TNS) — Phase 4
- No real KPU simulator integration — analytical models only
- No physical placement algorithm — area-based floorplan estimation

## Verification

1. `pytest tests/test_technology.py tests/test_kpu_config.py tests/test_floorplan.py tests/test_bandwidth.py -v` — foundation unit tests
2. `pytest tests/test_kpu_specialists.py tests/test_kpu_loop.py -v` — KPU micro-arch tests
3. `pytest tests/test_eda_tools.py tests/test_rtl_templates.py tests/test_rtl_loop.py tests/test_rtl_specialists.py -v` — RTL tests
4. `pytest tests/test_kpu_integration.py tests/test_rtl_integration.py -v` — integration tests (needs langgraph)
5. `pytest tests/test_soc_state.py tests/test_planner_dispatcher.py tests/test_specialists.py -v` — existing tests still pass
6. `python examples/demo_kpu_rtl.py` — Demo 4 shows KPU config → floorplan → bandwidth → RTL
7. `python examples/demo_soc_designer.py` — Demo 1 still works unchanged
8. `python examples/demo_soc_optimizer.py` — Demo 3 still works unchanged
