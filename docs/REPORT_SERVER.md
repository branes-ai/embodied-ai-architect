# Report Server - Usage Guide

## Overview

The Report Server is an **optional** component that provides a simple web interface for viewing generated reports. It is completely separate from the Orchestrator and follows clean separation of concerns.

## Architecture

```
ReportSynthesisAgent (Producer)
    ↓ writes to filesystem
./reports/{workflow_id}/
    ├── report.html
    ├── report.json
    ├── metadata.json
    └── assets/
        ├── hardware_comparison.png
        └── layer_distribution.png
    ↓ serves via HTTP
Report Server (Consumer)
    ↓
Web Browser
```

## Key Principles

1. **Separation of Concerns**:
   - ReportSynthesisAgent generates artifacts
   - Report Server serves artifacts
   - Orchestrator coordinates workflow (doesn't serve HTTP)

2. **Optional Component**:
   - Reports can be viewed directly by opening HTML files
   - Server is convenience, not requirement

3. **Static File Serving**:
   - No business logic in server
   - Can be replaced with nginx, Apache, or any web server

## Usage

### Start the Server

```bash
# From project root
python -m embodied_ai_architect.report_server

# Or specify custom port/host
python -m embodied_ai_architect.report_server --port 8080 --host 0.0.0.0
```

### Access Reports

1. **Index Page**: http://localhost:8000/
   - Lists all available reports
   - Provides links to HTML and JSON versions

2. **Specific Report**: http://localhost:8000/{workflow_id}/report.html
   - View individual workflow report

3. **JSON Data**: http://localhost:8000/{workflow_id}/report.json
   - Access machine-readable data

### Viewing Reports Without Server

Reports can be viewed directly without running the server:

```bash
# Open in browser directly
firefox reports/{workflow_id}/report.html

# Or use Python's built-in server
cd reports
python -m http.server 8000

# Or any other web server
cd reports
npx serve
```

## Report Structure

Each report consists of:

```
reports/{workflow_id}/
├── report.html          # Human-readable HTML report
├── report.json          # Machine-readable JSON data
├── metadata.json        # Workflow metadata
└── assets/              # Generated charts and visualizations
    ├── hardware_comparison.png
    ├── layer_distribution.png
    └── benchmark_comparison.png
```

## Integration Examples

### CLI Tool

```python
# View latest report
import subprocess
from pathlib import Path

reports = sorted(Path("reports").iterdir(), reverse=True)
latest = reports[0] / "report.html"
subprocess.run(["firefox", str(latest)])
```

### Jupyter Notebook

```python
# Load report data for analysis
import json
from pathlib import Path

report_path = Path("reports") / "30bb66aa" / "report.json"
data = json.loads(report_path.read_text())

print(f"Model: {data['model_analysis']['model_type']}")
print(f"Parameters: {data['model_analysis']['total_parameters']:,}")
print(f"Best Hardware: {data['hardware_recommendations'][0]['name']}")
```

### CI/CD Pipeline

```bash
# Run workflow and extract metrics
python run_workflow.py --model my_model.pt

# Parse JSON for CI metrics
python -c "
import json
from pathlib import Path

reports = sorted(Path('reports').iterdir(), reverse=True)
data = json.loads((reports[0] / 'report.json').read_text())

latency = data['benchmarks']['local_cpu']['mean_latency_ms']
print(f'latency_ms={latency}')

if latency > 100:
    exit(1)  # Fail if latency too high
"
```

### Email/Slack Integration

```python
# Send report link via Slack
import requests

workflow_id = "30bb66aa"
report_url = f"http://reports.mycompany.com/{workflow_id}/report.html"

slack_webhook = "https://hooks.slack.com/..."
requests.post(slack_webhook, json={
    "text": f"New Embodied AI report available: {report_url}"
})
```

## Production Deployment

### Using nginx

```nginx
server {
    listen 80;
    server_name reports.mycompany.com;

    root /path/to/embodied-ai-architect/reports;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
        autoindex on;  # Enable directory listing
    }

    # Cache static assets
    location ~* \.(png|jpg|jpeg|svg|css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### Using Docker

```dockerfile
FROM nginx:alpine
COPY reports/ /usr/share/nginx/html/
EXPOSE 80
```

```bash
docker build -t embodied-ai-reports .
docker run -p 8000:80 embodied-ai-reports
```

### Using Cloud Storage

```bash
# Sync reports to S3
aws s3 sync ./reports s3://my-bucket/reports/ --acl public-read

# Access via CloudFront CDN
https://d1234abcd.cloudfront.net/{workflow_id}/report.html
```

## API for Programmatic Access

The JSON reports can be accessed programmatically:

```bash
# List all reports
curl http://localhost:8000/

# Get specific report data
curl http://localhost:8000/30bb66aa/report.json | jq .

# Extract specific metrics
curl http://localhost:8000/30bb66aa/report.json | \
  jq '.hardware_recommendations[0].name'
```

## Security Considerations

1. **No Authentication**: The simple server has no auth. For production:
   - Use nginx with HTTP basic auth
   - Deploy behind corporate VPN
   - Use cloud provider's IAM policies

2. **Read-Only**: Server only reads files, never writes

3. **No Code Execution**: Pure static file serving

## Alternatives

You don't need the included server. Any of these work:

```bash
# Python built-in
python -m http.server 8000

# Node.js serve
npx serve reports

# PHP
php -S localhost:8000 -t reports

# Ruby
ruby -run -e httpd reports -p 8000
```

## Future Enhancements

Possible additions (not currently implemented):

- Search and filter reports
- Compare two reports side-by-side
- Real-time updates via WebSocket
- Export to PDF
- Authentication/authorization
- Database backend for metadata
- REST API for report management
