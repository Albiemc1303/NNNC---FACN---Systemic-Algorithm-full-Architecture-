"""
Provenance Explorer Dashboard - Core Components
Displays project metrics, narrative summaries, and provenance visualization.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

def load_provenance_records():
    """Load all provenance records from .provenance directory"""
    records = []
    prov_dir = Path(".provenance")
    for f in prov_dir.glob("PROVENANCE*.json"):
        with open(f) as fh:
            records.append(json.load(fh))
    return records

def calculate_metrics(records):
    """Calculate project metrics from provenance records"""
    metrics = {
        "total_commits": len(records),
        "unique_authors": len({r["author"] for r in records}),
        "artifacts_tracked": sum(len(r.get("artifacts", [])) for r in records),
        "last_update": max(r["timestamp"] for r in records),
    }
    return metrics

def generate_narrative(record):
    """Generate human-readable story from provenance record"""
    ts = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
    author = record["author"]["id"] if isinstance(record["author"], dict) else record["author"]
    
    story = f"On {ts.strftime('%B %d, %Y at %H:%M UTC')}, {author} "
    
    if record["action"] == "commit":
        story += f"committed changes affecting {len(record.get('artifacts', []))} files."
    elif record["action"] == "project_init":
        story += "initialized the project structure."
    else:
        story += f"performed a {record['action']} operation."
        
    return story

def draw_timeline(records):
    """Create interactive timeline visualization"""
    df = pd.DataFrame([
        {
            "timestamp": datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")),
            "action": r["action"],
            "author": r["author"]["id"] if isinstance(r["author"], dict) else r["author"],
            "artifacts": len(r.get("artifacts", []))
        }
        for r in records
    ])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.timestamp,
        y=df.artifacts,
        mode="markers+lines",
        name="Artifacts Changed",
        hovertemplate="%{x}<br>%{text}",
        text=df.apply(lambda r: f"Author: {r.author}<br>Action: {r.action}<br>Files: {r.artifacts}", axis=1)
    ))
    
    fig.update_layout(
        title="Project Timeline",
        xaxis_title="Time",
        yaxis_title="Files Changed"
    )
    
    return fig

def main():
    st.set_page_config(page_title="Provenance Explorer", layout="wide")
    st.title("🔍 Project Provenance Explorer")
    
    # Load data
    try:
        records = load_provenance_records()
        metrics = calculate_metrics(records)
    except Exception as e:
        st.error(f"Error loading provenance records: {e}")
        return
        
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Commits", metrics["total_commits"])
    with col2:
        st.metric("Unique Contributors", metrics["unique_authors"])
    with col3:
        st.metric("Artifacts Tracked", metrics["artifacts_tracked"])
    with col4:
        st.metric("Last Update", metrics["last_update"])
        
    # Timeline visualization
    st.subheader("Project Timeline")
    fig = draw_timeline(records)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity narrative
    st.subheader("Recent Activity")
    for record in sorted(records, key=lambda r: r["timestamp"], reverse=True)[:5]:
        with st.expander(f"{record['action']} - {record['timestamp']}"):
            st.write(generate_narrative(record))
            if "artifacts" in record:
                st.json(record["artifacts"])

if __name__ == "__main__":
    main()