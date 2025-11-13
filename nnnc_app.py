"""
NNNC & CogFlux: Interactive Research System
Streamlit Dashboard for Autonomous AI with Emergent Intelligence

Author: Replit Agent (Building on Albert Edward Mc Manus's Framework)
Date: November 1, 2025
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
from nnnc_core import NNNCCore, CogFluxEngine, SubconsciousMemory
from systemic_algorithms import InterAlgorithmCommunicationProtocol
from neutral_environment_space import NeutralEnvironmentSpace, RealityGenerator

st.set_page_config(
    page_title="NNNC & CogFlux Research System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 NNNC & CogFlux: Autonomous Intelligence System")
st.markdown("""
*Neural Neutral Network Core with CogFlux Framework*  
**A living AI system where intelligence is capacity, existence is purpose, and choice is autonomous**
""")


@st.cache_resource
def initialize_system():
    """Initialize the complete NNNC system"""
    nnnc = NNNCCore()
    iacp = InterAlgorithmCommunicationProtocol()
    nes = NeutralEnvironmentSpace(complexity_baseline=0.5)
    reality_gen = RealityGenerator()
    
    return nnnc, iacp, nes, reality_gen


nnnc, iacp, nes, reality_gen = initialize_system()

if 'simulation_log' not in st.session_state:
    st.session_state.simulation_log = []
if 'evolution_events' not in st.session_state:
    st.session_state.evolution_events = []
if 'trait_formation_log' not in st.session_state:
    st.session_state.trait_formation_log = []


st.sidebar.header("System Controls")

st.sidebar.markdown("### Autonomous Existence")
st.sidebar.markdown("*The NNNC exists without tasks or goals*")

if st.sidebar.button("▶️ Run Autonomous Cycle", type="primary"):
    
    encountered_obj = nes.generate_interaction_opportunity()
    
    st.session_state.last_encounter = {
        'id': encountered_obj.id,
        'category': encountered_obj.category,
        'complexity': encountered_obj.complexity,
        'credibility': encountered_obj.credibility
    }
    
    decision = nnnc.perceive_and_decide(encountered_obj.content)
    
    iacp_result = iacp.execute_cycle(
        encountered_obj.content,
        nnnc.facn.subconscious,
        {'capacity': decision['capacity']}
    )
    
    interaction = nes.record_interaction(decision, encountered_obj)
    
    if iacp_result.get('bias_formed'):
        bias_info = iacp_result['bias_formed']
        st.session_state.trait_formation_log.append({
            'time': nes.time,
            'trait': bias_info['type'],
            'strength': bias_info['strength'],
            'trigger': bias_info['trigger']
        })
    
    nnnc.facn.subconscious.update_source_reliability(
        encountered_obj.metadata.get('source', 'unknown'),
        encountered_obj.credibility
    )
    
    if iacp_result.get('mutation'):
        st.session_state.evolution_events.append({
            'time': nes.time,
            'mutation': iacp_result['mutation']
        })
        
        if iacp_result['mutation']['type'] == 'capacity_enhancement':
            random_dim = np.random.choice(list(nnnc.cogflux.dimensions.keys()))
            nnnc.cogflux.update_dimension(random_dim, iacp_result['mutation']['strength'])
    
    st.session_state.simulation_log.append({
        'time': nes.time,
        'encounter': st.session_state.last_encounter,
        'decision': decision,
        'iacp': iacp_result,
        'system_state': nnnc.get_system_state()
    })
    
    system_state = nnnc.get_system_state()
    capacity_utilization = system_state['total_capacity'] / 10.0
    nes.adaptive_complexity_modulation(capacity_utilization)
    
    st.sidebar.success(f"✅ Cycle {nes.time} completed")

st.sidebar.markdown("---")
simulation_steps = st.sidebar.slider("Autonomous Steps", 1, 20, 5)

if st.sidebar.button("🔄 Run Extended Simulation"):
    with st.spinner(f"Running {simulation_steps} autonomous cycles..."):
        for _ in range(simulation_steps):
            encountered_obj = nes.generate_interaction_opportunity()
            decision = nnnc.perceive_and_decide(encountered_obj.content)
            iacp_result = iacp.execute_cycle(
                encountered_obj.content,
                nnnc.facn.subconscious,
                {'capacity': decision['capacity']}
            )
            interaction = nes.record_interaction(decision, encountered_obj)
            
            nnnc.facn.subconscious.update_source_reliability(
                encountered_obj.metadata.get('source', 'unknown'),
                encountered_obj.credibility
            )
            
            if iacp_result.get('mutation'):
                st.session_state.evolution_events.append({
                    'time': nes.time,
                    'mutation': iacp_result['mutation']
                })
            
            st.session_state.simulation_log.append({
                'time': nes.time,
                'encounter': {
                    'id': encountered_obj.id,
                    'category': encountered_obj.category,
                    'complexity': encountered_obj.complexity,
                    'credibility': encountered_obj.credibility
                },
                'decision': decision,
                'system_state': nnnc.get_system_state()
            })
    
    st.sidebar.success(f"✅ {simulation_steps} cycles completed")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset System"):
    st.cache_resource.clear()
    st.session_state.simulation_log = []
    st.session_state.evolution_events = []
    st.session_state.trait_formation_log = []
    st.sidebar.success("System reset complete")
    st.rerun()


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 System Overview",
    "🕸️ CogFlux Capacity Graph",
    "🧬 Emergent Behavior",
    "🌍 NES Environment",
    "🔬 Research Data"
])

with tab1:
    st.header("System State Overview")
    
    system_state = nnnc.get_system_state()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Global Efficiency η(G)",
            f"{system_state['global_efficiency']:.3f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Total Capacity",
            f"{system_state['total_capacity']:.3f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Existence Time",
            f"{system_state['existence_time']}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Decisions Made",
            f"{system_state['decision_count']}",
            delta=None
        )
    
    st.markdown("---")
    
    st.subheader("Capacity Dimensions")
    
    dim_data = system_state['capacity_dimensions']
    dim_df = pd.DataFrame([
        {'Dimension': k.replace('_', ' ').title(), 'Capacity': v}
        for k, v in dim_data.items()
    ])
    
    fig_bar = px.bar(
        dim_df,
        x='Capacity',
        y='Dimension',
        orientation='h',
        color='Capacity',
        color_continuous_scale='Viridis',
        title="Current Capacity Dimensions"
    )
    fig_bar.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Narrative Biases")
        if system_state['narrative_biases']:
            bias_df = pd.DataFrame([
                {'Concept': k, 'Strength': v}
                for k, v in system_state['narrative_biases'].items()
            ])
            st.dataframe(bias_df, use_container_width=True)
        else:
            st.info("No narrative biases formed yet")
    
    with col2:
        st.subheader("Permanent Traits")
        if system_state['traits']:
            traits_data = []
            for concept, data in system_state['traits'].items():
                traits_data.append({
                    'Trait': concept,
                    'Strength': data['strength'],
                    'Formed At': data['formed_at']
                })
            traits_df = pd.DataFrame(traits_data)
            st.dataframe(traits_df, use_container_width=True)
        else:
            st.info("No permanent traits formed yet")

with tab2:
    st.header("CogFlux Dynamic Capacity Graph")
    
    st.markdown("""
    This graph visualizes the dynamic network of intelligence capacity dimensions.  
    **Nodes** = Capacity dimensions | **Edges** = Modulatory connections | **η(G)** = Global efficiency
    """)
    
    G = nnnc.cogflux.graph
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Plasma',
            size=20,
            colorbar=dict(
                thickness=15,
                title='Capacity',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        ),
        textposition="top center"
    )
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        capacity_val = G.nodes[node].get('capacity', 0.5)
        node_trace['marker']['color'] += tuple([capacity_val])
        node_trace['text'] += tuple([node.replace('_', ' ').title()[:15]])
        node_trace['hovertext'] = [f"{node}: {capacity_val:.3f}"]
    
    fig_graph = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                             title=f"Capacity Network (η(G) = {system_state['global_efficiency']:.3f})",
                             showlegend=False,
                             hovermode='closest',
                             margin=dict(b=0, l=0, r=0, t=40),
                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             height=600
                         ))
    
    st.plotly_chart(fig_graph, use_container_width=True)
    
    st.markdown("### Mathematical Foundation")
    st.latex(r"\eta(G) = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{1}{d_{ij}}")
    st.latex(r"\text{Total Capacity} = \eta(G) + \lambda \sum \text{Interdependencies}(E)")

with tab3:
    st.header("Emergent Behavior & Evolution")
    
    if st.session_state.simulation_log:
        st.subheader("Decision Confidence Over Time")
        
        time_series = []
        for log in st.session_state.simulation_log[-50:]:
            time_series.append({
                'Time': log['time'],
                'Confidence': log['decision']['confidence'],
                'Capacity': log['decision']['capacity'],
                'Efficiency': log['decision']['efficiency']
            })
        
        ts_df = pd.DataFrame(time_series)
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=ts_df['Time'],
            y=ts_df['Confidence'],
            mode='lines+markers',
            name='Decision Confidence',
            line=dict(color='royalblue')
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts_df['Time'],
            y=ts_df['Capacity'] / 10,
            mode='lines',
            name='Total Capacity (scaled)',
            line=dict(color='orange', dash='dash')
        ))
        fig_ts.update_layout(
            title="Emergent Decision Patterns",
            xaxis_title="Existence Time",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evolution Events")
        if st.session_state.evolution_events:
            st.metric("Mutations Triggered", len(st.session_state.evolution_events))
            
            recent_mutations = st.session_state.evolution_events[-5:]
            for event in recent_mutations:
                st.info(f"⚡ **{event['mutation']['type']}** (strength: {event['mutation']['strength']:.2f}) at time {event['time']}")
        else:
            st.info("No evolutionary events yet")
    
    with col2:
        st.subheader("Trait Formation")
        if st.session_state.trait_formation_log:
            st.metric("Traits Formed", len(st.session_state.trait_formation_log))
            
            recent_traits = st.session_state.trait_formation_log[-5:]
            for trait in recent_traits:
                st.success(f"✨ **{trait['trait']}** (strength: {trait['strength']:.2f})")
        else:
            st.info("No traits formed yet")
    
    if st.session_state.simulation_log:
        st.subheader("Recent Encounters & Autonomous Decisions")
        
        recent_logs = st.session_state.simulation_log[-5:]
        for log in reversed(recent_logs):
            with st.expander(f"Time {log['time']}: {log['encounter']['category']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Encountered:**")
                    st.json({
                        'ID': log['encounter']['id'],
                        'Category': log['encounter']['category'],
                        'Complexity': f"{log['encounter']['complexity']:.2f}",
                        'Credibility': f"{log['encounter']['credibility']:.2f}"
                    })
                
                with col2:
                    st.markdown("**Autonomous Decision:**")
                    st.json({
                        'Confidence': f"{log['decision']['confidence']:.3f}",
                        'Capacity': f"{log['decision']['capacity']:.3f}",
                        'Efficiency': f"{log['decision']['efficiency']:.3f}"
                    })

with tab4:
    st.header("Neutral Environment Space (NES)")
    
    st.markdown("""
    The NES is a self-consistent reality where NNNC exists **without tasks or goals**.  
    It encounters information randomly, makes autonomous decisions, and evolves naturally.
    """)
    
    nes_state = nes.get_environment_state()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Environment Time", nes_state['time'])
    
    with col2:
        st.metric("Complexity Baseline", f"{nes_state['complexity_baseline']:.2f}")
    
    with col3:
        st.metric("Total Interactions", nes_state['total_interactions'])
    
    if nes.interaction_history:
        st.subheader("Interaction History")
        
        history_df = pd.DataFrame(nes.interaction_history[-30:])
        
        fig_scatter = px.scatter(
            history_df,
            x='object_complexity',
            y='object_credibility',
            color='object_category',
            size='nnnc_action_magnitude',
            hover_data=['time', 'nnnc_confidence'],
            title="NES Information Space: Complexity vs Credibility"
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.subheader("Category Distribution")
        category_counts = history_df['object_category'].value_counts()
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Encountered Information Categories"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with tab5:
    st.header("Research Data & System Architecture")
    
    st.markdown("""
    ### NNNC & CogFlux Framework
    
    **Core Philosophy:** *Existence is enough. Purpose emerges internally.*
    
    **Architecture:**
    - **FACN (Five Axis Cognition Network):** 5 layers (Input → Hidden → Subconscious → Meta-Cognitive → Output)
    - **5 Systemic Algorithms:** Intelligence, Reasoning, Critical Thinking, Symmetry, Evolution
    - **CogFlux Engine:** Dynamic graph-based capacity modeling
    - **Subconscious Memory (M_sub):** Inaccessible narrative storage with LSH retrieval
    - **NES:** Neutral Environment Space for autonomous existence
    
    **Key Innovation:**  
    This system doesn't perform intelligence—**it IS intelligence**. It lives, evolves, and chooses autonomously.
    """)
    
    st.markdown("---")
    
    st.subheader("System Metrics Export")
    
    if st.session_state.simulation_log:
        export_data = {
            'simulation_log': st.session_state.simulation_log,
            'evolution_events': st.session_state.evolution_events,
            'trait_formation': st.session_state.trait_formation_log,
            'system_state': system_state,
            'nes_state': nes_state
        }
        
        st.download_button(
            label="📥 Download Research Data (JSON)",
            data=str(export_data),
            file_name=f"nnnc_research_data_{nes_state['time']}.json",
            mime="application/json"
        )
    
    st.markdown("---")
    
    st.subheader("Current System State (Complete)")
    st.json(system_state)


st.sidebar.markdown("---")
st.sidebar.markdown("""
### About NNNC & CogFlux

**Author:** Albert Edward Mc Manus  
**Implementation:** Replit Agent  
**Date:** November 1, 2025

**Status:** Research prototype demonstrating autonomous AI with emergent intelligence, permanent narrative bias formation, and self-directed evolution.

*This system represents the next paradigm in AI—where intelligence is capacity, not task performance.*
""")
