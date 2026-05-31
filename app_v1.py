"""
app.py - PakSignal: Adaptive Traffic Signal Control Dashboard

Run with:
    streamlit run app.py

Make sure best.pt is in the models/ folder (or set the path in the sidebar).
"""

import os
import sys
import random
import tempfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src/ is importable when running from the PakSignal directory
sys.path.insert(0, os.path.dirname(__file__))

from src.detector import analyze_feed
from src.pcu_logic import calculate_pcu_score, PCU_WEIGHTS
from src.signal_controller import (
    allocate_green_times,
    apply_emergency_priority,
    static_vs_adaptive_comparison,
)
from src.environment import estimate_environmental_impact

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="PakSignal",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark metric cards */
    .metric-card {
        background: #1e1e2e;
        padding: 1rem 1.25rem;
        border-radius: 0.6rem;
        border-left: 4px solid #7c3aed;
        margin-bottom: 0.75rem;
    }
    /* Emergency alert box */
    .emv-alert {
        background: #450a0a;
        border: 1px solid #dc2626;
        padding: 0.6rem 1rem;
        border-radius: 0.4rem;
        color: #fca5a5;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    /* Priority decision box */
    .priority-box {
        background: #052e16;
        border: 1px solid #16a34a;
        padding: 0.8rem 1.2rem;
        border-radius: 0.4rem;
        color: #86efac;
        font-weight: 700;
        font-size: 1.05rem;
    }
    /* Disclaimer text */
    .disclaimer {
        color: #9ca3af;
        font-size: 0.78rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🚦 PakSignal")
st.subheader("YOLOv8-Based Adaptive Traffic Signal Control for Pakistani Roads")
st.markdown(
    "Upload one traffic feed per direction. PakSignal detects Pakistani vehicles "
    "(bike, car, rickshaw, htv, emv), calculates **PCU-weighted** traffic load, "
    "and recommends **adaptive green signal timing** — with emergency vehicle priority."
)
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_path = st.text_input(
        "Model path (best.pt)",
        value="models/best.pt",
        help="Relative or absolute path to your trained YOLOv8 weights.",
    )

    demo_mode = st.checkbox(
        "Demo mode (simulate detections)",
        value=False,
        help=(
            "Tick this to run with simulated vehicle counts. "
            "Useful for showing the dashboard before training is complete."
        ),
    )

    st.divider()
    conf_threshold = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    max_frames     = st.slider("Max frames per video", 10, 120, 60, 10)
    static_green   = st.slider("Static timer baseline (s)", 15, 60, 30, 5)
    total_cycle    = st.slider("Total signal cycle (s)", 60, 240, 120, 10)
    min_green      = st.slider("Min green per lane (s)", 10, 30, 15, 5)
    max_green      = st.slider("Max green per lane (s)", 30, 90, 60, 5)

    st.divider()
    st.markdown("**PCU weights (IRC standard)**")
    for cls, w in PCU_WEIGHTS.items():
        st.markdown(f"- `{cls}`: {w}")

    st.divider()
    st.caption("Dataset: density-traffic-controller-v1 v8 (Roboflow)")
    st.caption("Classes: bike · car · emv · htv · rickshaw")


# ─── Feed Uploaders ───────────────────────────────────────────────────────────
st.header("📷 Four-Direction Traffic Feeds")
st.markdown(
    "Upload an **image** (jpg/png) or **video** (mp4/avi) for each direction. "
    "You can upload any subset — unused directions are skipped."
)

col_n, col_s, col_e, col_w = st.columns(4)
uploads = {}

with col_n:
    st.markdown("### ⬆️ North")
    uploads["North"] = st.file_uploader(
        "North feed", type=["jpg", "jpeg", "png", "mp4", "avi"], key="North"
    )
with col_s:
    st.markdown("### ⬇️ South")
    uploads["South"] = st.file_uploader(
        "South feed", type=["jpg", "jpeg", "png", "mp4", "avi"], key="South"
    )
with col_e:
    st.markdown("### ➡️ East")
    uploads["East"] = st.file_uploader(
        "East feed", type=["jpg", "jpeg", "png", "mp4", "avi"], key="East"
    )
with col_w:
    st.markdown("### ⬅️ West")
    uploads["West"] = st.file_uploader(
        "West feed", type=["jpg", "jpeg", "png", "mp4", "avi"], key="West"
    )

st.divider()

# ─── Analyze Button ───────────────────────────────────────────────────────────
analyze_clicked = st.button(
    "🔍  Analyze Traffic and Allocate Signal Timing",
    type="primary",
    use_container_width=True,
)

# ─── Demo helper ──────────────────────────────────────────────────────────────
def _demo_lane(direction: str) -> dict:
    """Generate a plausible simulated detection result for demo purposes."""
    counts = {
        "bike":     random.randint(0, 10),
        "car":      random.randint(0, 8),
        "rickshaw": random.randint(0, 4),
        "htv":      random.randint(0, 2),
        "emv":      random.randint(0, 1),
    }
    # Remove zero-count classes so the display is cleaner
    counts = {k: v for k, v in counts.items() if v > 0}
    total = sum(counts.values())
    pcu = calculate_pcu_score(counts)
    return {
        "direction":         direction,
        "raw_counts":        counts,
        "total_vehicles":    total,
        "frames_analyzed":   random.randint(30, 60),
        "detected_emergency": counts.get("emv", 0) > 0,
        "pcu_score":         pcu,
        "source_type":       "demo",
    }


# ─── Main Analysis Logic ──────────────────────────────────────────────────────
if analyze_clicked:

    active_feeds = {d: f for d, f in uploads.items() if f is not None}

    # ── Demo mode: ignore uploads and simulate data ───────────────────────────
    if demo_mode:
        st.info("Running in **Demo Mode** — vehicle counts are simulated.")
        lane_results = [_demo_lane(d) for d in ["North", "South", "East", "West"]]

    # ── Real mode: run YOLO on uploaded feeds ─────────────────────────────────
    else:
        if not active_feeds:
            st.warning("Please upload at least one traffic feed, or enable Demo Mode.")
            st.stop()

        if not os.path.isfile(model_path):
            st.error(
                f"Model file not found at `{model_path}`. "
                "Train the model first and update the path in the sidebar, "
                "or enable **Demo Mode** to see the dashboard without a model."
            )
            st.stop()

        lane_results = []
        progress_bar = st.progress(0, text="Starting analysis...")

        for i, (direction, uploaded_file) in enumerate(active_feeds.items()):
            progress_bar.progress(i / len(active_feeds), text=f"Analyzing {direction}...")

            # Save uploaded bytes to a temp file so OpenCV / YOLO can read it
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                result = analyze_feed(
                    source_path=tmp_path,
                    direction_name=direction,
                    model_path=model_path,
                    conf=conf_threshold,
                    max_frames=max_frames,
                )
                lane_results.append(result)
            except Exception as exc:
                st.error(f"Error analyzing **{direction}** feed: {exc}")
            finally:
                os.unlink(tmp_path)

        progress_bar.progress(1.0, text="Detection complete!")

        if not lane_results:
            st.error("No feeds could be analyzed. Check your files and model path.")
            st.stop()

    # ── Signal allocation ─────────────────────────────────────────────────────
    lane_results = allocate_green_times(
        lane_results,
        total_cycle_time=total_cycle,
        min_green=min_green,
        max_green=max_green,
    )
    lane_results = apply_emergency_priority(lane_results)
    lane_results = static_vs_adaptive_comparison(lane_results, static_green=static_green)
    env_impact   = estimate_environmental_impact(lane_results, static_green=static_green)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: Per-lane cards
    # ═══════════════════════════════════════════════════════════════════════════
    st.header("📊 Lane-by-Lane Analysis")
    lane_cols = st.columns(len(lane_results))

    for col, lane in zip(lane_cols, lane_results):
        with col:
            icon = "🚨" if lane["detected_emergency"] else "🚗"
            st.subheader(f"{icon} {lane['direction']}")

            if lane["detected_emergency"]:
                st.markdown(
                    '<div class="emv-alert">⚠️ EMERGENCY VEHICLE DETECTED</div>',
                    unsafe_allow_html=True,
                )

            st.metric("Vehicles detected", lane["total_vehicles"])
            st.metric("PCU score", f"{lane['pcu_score']:.1f}")
            st.metric("Adaptive green", f"{lane['green_time']} s")
            st.metric("Static baseline", f"{lane['static_green']} s")
            diff = lane["green_time_diff"]
            diff_label = f"+{diff}s" if diff >= 0 else f"{diff}s"
            st.caption(f"Difference vs static: **{diff_label}**")
            st.caption(f"Frames analyzed: {lane['frames_analyzed']}")

            # Vehicle class breakdown
            if lane["raw_counts"]:
                st.markdown("**By class:**")
                for cls, cnt in sorted(lane["raw_counts"].items()):
                    pcu_w = PCU_WEIGHTS.get(cls, 1.0)
                    st.markdown(f"&nbsp;&nbsp;• `{cls}`: {cnt} × {pcu_w} PCU")
            else:
                st.markdown("*No vehicles detected*")

            if lane.get("priority_reason"):
                st.caption(f"Priority: {lane['priority_reason']}")
            else:
                st.caption(f"Allocation: {lane.get('allocation_reason', '')}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: Charts
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.header("📈 Traffic Visualisation")

    directions   = [l["direction"]    for l in lane_results]
    pcu_scores   = [l["pcu_score"]    for l in lane_results]
    green_times  = [l["green_time"]   for l in lane_results]

    chart_left, chart_right = st.columns(2)

    with chart_left:
        fig_pcu = px.bar(
            x=directions, y=pcu_scores,
            title="PCU Score per Lane (higher = more traffic)",
            labels={"x": "Direction", "y": "PCU Score"},
            color=pcu_scores,
            color_continuous_scale="Reds",
            text_auto=True,
        )
        fig_pcu.update_layout(coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig_pcu, use_container_width=True)

    with chart_right:
        fig_green = go.Figure()
        fig_green.add_trace(go.Bar(
            name="Adaptive", x=directions, y=green_times,
            marker_color="#22c55e", text=green_times, textposition="outside",
        ))
        fig_green.add_trace(go.Bar(
            name=f"Static ({static_green}s)",
            x=directions,
            y=[static_green] * len(lane_results),
            marker_color="#6b7280",
        ))
        fig_green.update_layout(
            barmode="group",
            title="Green Time: Adaptive vs Static (seconds)",
            xaxis_title="Direction",
            yaxis_title="Seconds",
        )
        st.plotly_chart(fig_green, use_container_width=True)

    # Vehicle distribution stacked bar
    all_classes = set()
    for lane in lane_results:
        all_classes.update(lane["raw_counts"].keys())

    if all_classes:
        dist_data = {"Direction": directions}
        for cls in sorted(all_classes):
            dist_data[cls] = [lane["raw_counts"].get(cls, 0) for lane in lane_results]

        dist_df = pd.DataFrame(dist_data)
        fig_dist = px.bar(
            dist_df, x="Direction", y=sorted(all_classes),
            title="Vehicle Class Distribution per Lane",
            barmode="stack",
            labels={"value": "Count", "variable": "Class"},
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Environmental impact
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.header("🌿 Estimated Environmental & Economic Impact")

    st.markdown(
        f'<p class="disclaimer">{env_impact["note"]}</p>',
        unsafe_allow_html=True,
    )

    e1, e2, e3, e4 = st.columns(4)
    e1.metric(
        "⏱️ Total wait reduced",
        f"{env_impact['estimated_wait_reduction_sec']:.0f} s",
        help="Vehicle-seconds of idle waiting saved across all lanes",
    )
    e2.metric(
        "⛽ Fuel saved",
        f"{env_impact['fuel_saved_liters']:.4f} L",
        help="Estimated petrol not burned due to less idling",
    )
    e3.metric(
        "💨 CO₂ reduced",
        f"{env_impact['co2_saved_g']:.1f} g",
        help="CO2 not emitted (2300 g per liter of petrol)",
    )
    e4.metric(
        "💰 PKR saved",
        f"Rs. {env_impact['pkr_saved']:.2f}",
        help="Cost of fuel saved at 270 PKR/L",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 4: Final signal decision
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.header("🏆 Signal Decision")

    emv_lanes = [l for l in lane_results if l.get("detected_emergency")]
    if emv_lanes:
        priority_lane = emv_lanes[0]
        reason = f"Emergency vehicle (EMV) detected in {priority_lane['direction']}"
    else:
        priority_lane = max(lane_results, key=lambda l: l["pcu_score"])
        reason = f"Highest PCU score ({priority_lane['pcu_score']:.1f}) among all lanes"

    st.markdown(
        f'<div class="priority-box">'
        f"🟢 <b>Priority lane: {priority_lane['direction']}</b><br>"
        f"Reason: {reason}<br>"
        f"Recommended green time: <b>{priority_lane['green_time']} seconds</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 5: Export
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.header("💾 Export Results")

    export_rows = []
    for lane in lane_results:
        row = {
            "direction":           lane["direction"],
            "total_vehicles":      lane["total_vehicles"],
            "pcu_score":           lane["pcu_score"],
            "green_time_adaptive": lane["green_time"],
            "green_time_static":   lane["static_green"],
            "green_time_diff":     lane["green_time_diff"],
            "detected_emergency":  lane["detected_emergency"],
            "priority_reason":     lane.get("priority_reason", ""),
            "allocation_reason":   lane.get("allocation_reason", ""),
            "frames_analyzed":     lane["frames_analyzed"],
            "source_type":         lane.get("source_type", ""),
        }
        for cls, cnt in lane["raw_counts"].items():
            row[f"count_{cls}"] = cnt
        export_rows.append(row)

    results_df = pd.DataFrame(export_rows)
    csv_bytes   = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download lane_results.csv",
        data=csv_bytes,
        file_name="lane_results.csv",
        mime="text/csv",
    )
    st.dataframe(results_df, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p class="disclaimer">'
    "PakSignal — Semester Deep Learning Project. "
    "Perception: YOLOv8n trained on density-traffic-controller-v1 (Roboflow). "
    "Signal logic: PCU-weighted rule-based allocation. "
    "DQN/RL integration: future work."
    "</p>",
    unsafe_allow_html=True,
)
