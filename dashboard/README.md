# Provenance Explorer Dashboard

This dashboard provides a comprehensive view of the project's provenance, integrity, and development history.

## Features

- **Project Metrics**
  - Total commits and changes
  - Unique contributors
  - Artifacts tracked
  - Latest updates
  
- **Timeline Visualization**
  - Interactive timeline of all project events
  - Artifact change tracking
  - Contributor activity
  
- **Narrative Summaries**
  - Human-readable event descriptions
  - Recent activity log
  - Detailed provenance records

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run dashboard/app.py
```

## Development

To extend the dashboard:
- Add new metrics in calculate_metrics()
- Enhance narrative generation in generate_narrative()
- Create new visualizations in draw_timeline()