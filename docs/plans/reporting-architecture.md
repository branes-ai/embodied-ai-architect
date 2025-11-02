# Reporting Architecture - Separation of Concerns

**Date**: 2025-11-02
**Status**: Design Proposal

## Problem Statement

The ReportSynthesisAgent needs to generate visualizations and reports from workflow results. However:
- The Orchestrator should not be responsible for rendering views or serving HTTP
- Reports should be accessible independently of the workflow execution
- Multiple consumers might need access to reports (web UI, CLI, API clients)
- Reports should be persistent and shareable

## Architectural Principles

### 1. Separation of Concerns
- **Orchestrator**: Coordinates agent workflows, manages state
- **ReportAgent**: Transforms data → generates artifacts (HTML, JSON, images)
- **Report Server**: Optional separate service for viewing/serving reports
- **Storage Layer**: Persistent storage for report artifacts

### 2. Producer-Consumer Pattern
```
ReportAgent (Producer)
    ↓ generates
Report Artifacts (Storage)
    ↓ reads
Report Server (Consumer)
```

### 3. Multiple Output Formats
- **Machine-readable**: JSON for programmatic access
- **Human-readable**: HTML for web viewing
- **Visualizations**: PNG/SVG charts embedded in HTML
- **Raw data**: CSV/Parquet for further analysis

## Proposed Architecture

### Option 1: Filesystem-Based (Recommended for MVP)

```
Workflow Execution
    ↓
ReportAgent.execute()
    ↓ generates artifacts
./reports/
    ├── {workflow_id}/
    │   ├── report.json          # Machine-readable data
    │   ├── report.html          # Human-readable report
    │   ├── assets/
    │   │   ├── hardware_comparison.png
    │   │   ├── performance_chart.png
    │   │   └── architecture_diagram.png
    │   └── metadata.json        # Workflow metadata
    ↓
Optional: Simple HTTP Server
    - Serves static files from ./reports/
    - No business logic
    - Can be nginx, Python http.server, or custom FastAPI
```

**Benefits**:
- Simple, no infrastructure dependencies
- Reports persist across runs
- Easy to share (copy directory)
- Can be served by any web server
- Testable without server

**Implementation**:
```python
class ReportSynthesisAgent(BaseAgent):
    def execute(self, input_data):
        # Generate report data
        report_data = self._synthesize_results(input_data)

        # Generate visualizations
        charts = self._create_visualizations(report_data)

        # Write artifacts to disk
        report_dir = Path(f"./reports/{workflow_id}")
        report_dir.mkdir(parents=True, exist_ok=True)

        # Machine-readable
        (report_dir / "report.json").write_text(json.dumps(report_data))

        # Human-readable
        html = self._generate_html(report_data, charts)
        (report_dir / "report.html").write_text(html)

        # Return reference, not full HTML
        return AgentResult(
            success=True,
            data={
                "report_path": str(report_dir),
                "report_url": f"/reports/{workflow_id}/report.html",
                "summary": report_data["summary"]
            }
        )
```

### Option 2: Database-Backed

```
ReportAgent
    ↓ writes
Database (PostgreSQL/MongoDB)
    ├── reports table
    ├── visualizations (binary/blob)
    └── workflow_history
    ↓ reads
Report API Server (FastAPI)
    ↓ serves
Web UI / API Clients
```

**Benefits**:
- Queryable report history
- Better for large-scale deployments
- Enables analytics across reports
- Role-based access control

**Drawbacks**:
- Infrastructure dependency
- More complex setup
- Harder to share individual reports

### Option 3: Hybrid (Object Storage)

```
ReportAgent
    ↓ uploads
Object Storage (S3, MinIO, GCS)
    └── bucket/reports/{workflow_id}/
    ↓ generates presigned URLs
Clients access directly
```

**Benefits**:
- Scalable storage
- CDN integration
- Cloud-native
- Easy sharing via URLs

## Recommended Approach: Filesystem + Optional Server

### Phase 1: Filesystem-Based Reports
1. ReportAgent generates artifacts to `./reports/{workflow_id}/`
2. Returns reference path in AgentResult
3. Orchestrator logs the report location
4. Users can open HTML directly or use CLI to view

### Phase 2: Simple Report Server (Optional)
```python
# reports_server.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# GET /reports/{workflow_id}/report.html
# GET /reports/latest → symlink to latest
# GET /api/reports → list all reports (JSON)
```

Start server independently:
```bash
python -m embodied_ai_architect.report_server
# or
uvicorn embodied_ai_architect.report_server:app
```

### Phase 3: Enhanced Server (Future)
- Database backend for report metadata
- Search/filter reports
- Comparison views (diff two reports)
- Real-time updates via WebSocket
- Authentication/authorization

## Report Agent Design

### Responsibilities
1. **Data Aggregation**: Collect results from all agents
2. **Synthesis**: Calculate derived metrics (speedup, cost-effectiveness)
3. **Visualization**: Generate charts (matplotlib, plotly)
4. **Template Rendering**: Create HTML reports (Jinja2)
5. **Artifact Management**: Write to storage

### NOT Responsibilities
- ❌ Serving HTTP requests
- ❌ User authentication
- ❌ Real-time updates
- ❌ Data querying (that's storage layer)

## Data Flow

```
Orchestrator.process(request)
    ↓
agent_results = {
    "ModelAnalyzer": {...},
    "HardwareProfile": {...},
    "Benchmark": {...}
}
    ↓
ReportAgent.execute({
    "workflow_id": uuid,
    "agent_results": agent_results,
    "request": original_request,
    "timestamp": datetime.now()
})
    ↓
Report Generated
    ├── ./reports/{workflow_id}/report.html
    ├── ./reports/{workflow_id}/report.json
    └── ./reports/{workflow_id}/assets/*.png
    ↓
Orchestrator returns:
    WorkflowResult(
        ...,
        report_path="./reports/{workflow_id}"
    )
```

## Report Structure

### JSON Format
```json
{
  "metadata": {
    "workflow_id": "abc-123",
    "timestamp": "2025-11-02T10:30:00Z",
    "model_name": "SimpleCNN",
    "duration_seconds": 45.2
  },
  "model_analysis": {
    "parameters": 620810,
    "layers": 19,
    "memory_mb": 2.4
  },
  "hardware_recommendations": [
    {
      "rank": 1,
      "name": "NVIDIA Jetson AGX Orin",
      "score": 91.4,
      "cost_usd": 2000,
      "power_watts": 60
    }
  ],
  "benchmarks": {
    "local_cpu": {
      "mean_latency_ms": 0.545,
      "throughput": 1835.29
    }
  },
  "insights": {
    "best_hardware": "NVIDIA Jetson AGX Orin",
    "cost_per_inference": 0.0001,
    "meets_constraints": true
  }
}
```

### HTML Structure
```html
<!DOCTYPE html>
<html>
<head>
    <title>Embodied AI Report - {workflow_id}</title>
    <style>/* Embedded CSS */</style>
</head>
<body>
    <header>
        <h1>Embodied AI Architect Report</h1>
        <div class="metadata">...</div>
    </header>

    <section id="executive-summary">
        <h2>Executive Summary</h2>
        <div class="key-metrics">...</div>
    </section>

    <section id="model-analysis">
        <h2>Model Analysis</h2>
        <table>...</table>
    </section>

    <section id="hardware-recommendations">
        <h2>Hardware Recommendations</h2>
        <img src="assets/hardware_comparison.png">
        <table>...</table>
    </section>

    <section id="benchmarks">
        <h2>Benchmark Results</h2>
        <img src="assets/performance_chart.png">
    </section>

    <section id="recommendations">
        <h2>Recommendations</h2>
        <ul>...</ul>
    </section>
</body>
</html>
```

## Advantages of This Architecture

### 1. Loose Coupling
- Orchestrator doesn't know about HTML/visualization
- Report server doesn't know about workflows
- Each component has single responsibility

### 2. Testability
- ReportAgent can be tested by checking file output
- No need to mock HTTP servers
- Easy to verify report structure

### 3. Flexibility
- Reports can be viewed offline (no server needed)
- Easy to archive/share reports (just copy directory)
- Can switch servers without changing report generation

### 4. Scalability
- Filesystem → NFS/shared storage for distributed systems
- Add CDN in front of report server
- Reports are immutable (easy to cache)

### 5. Multiple Consumers
```
Report Artifacts
    ├─→ Web Browser (via HTTP)
    ├─→ CLI Tool (read JSON)
    ├─→ Jupyter Notebook (load JSON)
    ├─→ CI/CD Pipeline (parse metrics)
    └─→ Email/Slack (attach HTML)
```

## Implementation Plan

1. **ReportSynthesisAgent** (Core)
   - Data aggregation and synthesis
   - Matplotlib/Plotly for charts
   - Jinja2 for HTML templates
   - Write to `./reports/`

2. **Report Viewer CLI** (Optional)
   - `embodied-ai-architect view-report {workflow_id}`
   - Opens HTML in browser
   - Or prints JSON summary

3. **Report Server** (Optional)
   - Simple FastAPI app
   - Serves static files
   - Optional API endpoints for listing reports

4. **Report Utilities** (Helper)
   - `list_reports()` - Find all reports
   - `compare_reports()` - Diff two workflows
   - `export_report()` - Convert formats

## Future Enhancements

- **Live Dashboard**: WebSocket for real-time updates during long benchmarks
- **Report Comparison**: Side-by-side diff of two workflows
- **Export Formats**: PDF generation, PowerPoint slides
- **Collaboration**: Share reports with annotations
- **Alerts**: Email/Slack when workflow completes

## Conclusion

**Recommendation**: Implement filesystem-based report generation with optional HTTP server.

This provides:
- ✅ Clean separation of concerns
- ✅ Simple deployment (no mandatory server)
- ✅ Easy testing and debugging
- ✅ Flexibility for future enhancements
- ✅ Offline access to reports
