"""
src/dashboard.py
BotnetLens — Streamlit Interactive Dashboard

Pages:
  1. 🏠 Overview          - project intro, pipeline architecture
  2. 🔌 Hardware Setup    - network topology diagram, device list, cost
  3. 🧪 Injection Demo    - slider 0-100%, live charts, cascade breakdown
  4. 🤖 Model Results     - binary, multiclass, anomaly comparison
  5. 📊 Charts            - all generated charts from reports/
  6. 📋 Full Report       - complete results table, key findings

Run:
    .venv\Scripts\streamlit run src/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
REPORTS      = ROOT / "reports"
INJ_DIR      = REPORTS / "injection"
FIG_DIR      = REPORTS / "figures"
RESULTS_CSV  = INJ_DIR / "injection_sweep_results.csv"

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BotnetLens — CTU-13 Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar navigation ─────────────────────────────────────────────────────
st.sidebar.title("🔍 BotnetLens")
st.sidebar.caption("CTU-13 Botnet Detection System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🔌 Hardware Setup",
        "🧪 Injection Demo",
        "🤖 Model Results",
        "📊 Charts",
        "📋 Full Report",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Status:** Software Simulation ✅\n\n"
    "**Hardware:** Pending 🔲\n\n"
    "**Models:** LightGBM + IsoForest"
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper — load results
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_CSV)
    df["tpr_pct"] = (df["tpr"]  * 100).round(2)
    df["fpr_pct"] = (df["fpr"]  * 100).round(2)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🔍 BotnetLens — CTU-13 Botnet Detection")
    st.caption("Major Project | Two-Phase Cascade Detection with Zero-Day Identification")

    st.markdown("---")

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Flows",      "19.97 M",  "CTU-13 real traffic")
    c2.metric("Binary Accuracy",    "91.07 %",  "LightGBM")
    c3.metric("Family Classes",     "7",         "6 known + 1 zero-day")
    c4.metric("Inference Latency",  "0.044 ms", "per flow (real-time)")

    st.markdown("---")
    st.subheader("📐 System Architecture")

    # Pipeline diagram using matplotlib
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.axis("off")
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    boxes = [
        (0.3,  1.2, 1.8, "CTU-13\nDataset\n19.97M flows", "#1f77b4"),
        (2.5,  1.2, 1.8, "Feature\nEngine   ering\n18 UBF features", "#2ca02c"),
        (4.7,  1.2, 1.8, "Phase 1\nBinary\nLightGBM 91%",  "#ff7f0e"),
        (6.9,  1.2, 1.8, "Phase 1\nMulticlass\nLightGBM 72%",  "#9467bd"),
        (9.1,  1.2, 1.8, "Phase 2\nAnomaly\nLOF 35.9%",  "#d62728"),
        (11.3, 1.2, 1.4, "CASCADE\nOUTPUT",              "#17becf"),
    ]

    for x, y, w, label, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, 1.6, boxstyle="round,pad=0.1",
            linewidth=1.5, edgecolor=color, facecolor=color + "33"
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + 0.8, label, ha="center", va="center",
                color="white", fontsize=7.5, fontweight="bold")

    # Arrows
    arrow_xs = [2.1, 4.3, 6.5, 8.7, 10.9]
    for ax_x in arrow_xs:
        ax.annotate("", xy=(ax_x + 0.4, 2.0), xytext=(ax_x, 2.0),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

    # Labels under arrows
    ax.text(4.4,  0.7, "Botnet?", ha="center", color="#ff7f0e", fontsize=7)
    ax.text(6.6,  0.7, "Family?", ha="center", color="#9467bd", fontsize=7)
    ax.text(8.8,  0.7, "Novel?",  ha="center", color="#d62728", fontsize=7)

    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.subheader("📂 Dataset — CTU-13 Botnet Families")

    family_data = {
        "Family":   ["Neris ⭐", "Rbot", "Virut", "Murlo", "Menti", "NsisAy 🔒", "Sogou"],
        "Flows":    [246889, 143918, 40904, 6127, 4630, 2168, 63],
        "Scenarios":["1,2,9", "3,4,10,11", "5,13", "8", "6", "12", "7"],
        "C&C":      ["IRC 6667", "IRC", "IRC/HTTP", "HTTP", "IRC", "ZERO-DAY", "HTTP"],
        "Role":     ["Injection family", "Training", "Training", "Training",
                     "Training", "Zero-day holdout", "Training"],
    }
    fdf = pd.DataFrame(family_data)
    st.dataframe(fdf, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Hardware Setup
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔌 Hardware Setup":
    st.title("🔌 Hardware Testbed Architecture")

    st.info(
        "**Current Status:** Running in SOFTWARE SIMULATION mode ✅ \n\n"
        "Physical hardware testbed is the next step after obtaining the required devices."
    )

    st.markdown("---")
    st.subheader("🗺️ Network Topology")

    # Topology diagram
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    def draw_device(ax, x, y, w, h, label, sublabel, color):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            linewidth=2, edgecolor=color, facecolor=color + "22"
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, label,    ha="center", va="center",
                color=color,   fontsize=9, fontweight="bold")
        ax.text(x + w/2, y + h*0.25, sublabel, ha="center", va="center",
                color="white", fontsize=7)

    # Devices
    draw_device(ax, 0.3, 1.8, 2.4, 2.4,
                "🖥️ Injector PC",
                "Scapy replays\nNeris binetflow\n10%→100% injection",
                "#1f77b4")
    draw_device(ax, 4.8, 1.8, 2.4, 2.4,
                "🔀 Managed Switch",
                "TP-Link TL-SG108E\nPort Mirroring ON\n~₹2,000",
                "#ff7f0e")
    draw_device(ax, 9.3, 1.8, 2.4, 2.4,
                "🖥️ Monitor PC",
                "realtime_sniffer.py\ncascade_inference.py\nhardware_monitor.py",
                "#2ca02c")

    # Ethernet cable — injector to switch
    ax.annotate("", xy=(4.8, 3.0), xytext=(2.7, 3.0),
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=3))
    ax.text(3.75, 3.3, "Ethernet", ha="center", color="#aaaaaa", fontsize=8)

    # Port mirror arrow — switch to monitor
    ax.annotate("", xy=(9.3, 3.0), xytext=(7.2, 3.0),
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=3))
    ax.text(8.25, 3.3, "Port Mirror", ha="center", color="#ff7f0e", fontsize=8)

    # USB NIC labels
    ax.text(0.5, 1.5, "USB Ethernet NIC", color="#888888", fontsize=7)
    ax.text(9.3, 1.5, "USB Ethernet NIC", color="#888888", fontsize=7)

    # Software flow
    ax.text(6.0, 0.5,
            "realtime_sniffer.py  →  cascade_inference.py  →  hardware_monitor.py  →  ALERTS",
            ha="center", color="#17becf", fontsize=8)

    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Required Devices")
        devices = {
            "Device": [
                "Injector PC (your laptop)",
                "Monitor PC (your laptop)",
                "TP-Link TL-SG108E Switch",
                "USB Gigabit Ethernet × 2",
                "Cat5e/6 Ethernet cables × 3",
                "Raspberry Pi 4 (optional)",
            ],
            "Status": ["✅ Owned", "✅ Owned", "🔲 Buy", "🔲 Buy", "🔲 Buy", "🔲 Optional"],
            "Cost":   ["₹0", "₹0", "₹1,500–2,500", "₹600 each", "₹200 total", "₹4,500–6,000"],
        }
        st.dataframe(pd.DataFrame(devices), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("🔧 Connection Steps")
        st.markdown("""
1. **Connect** Injector PC → Switch (Port 1) via Ethernet
2. **Connect** Monitor PC → Switch (Port 2) via Ethernet
3. **Enable** port mirroring: Port 1 → Port 2 (in switch web UI)
4. **Run on Monitor PC:**
```bash
python src/realtime_sniffer.py --iface "Ethernet"
python src/hardware_monitor.py
```
5. **Run on Injector PC:**
```bash
python src/inject_experiment.py --total-flows 10000
```
6. **Observe** live detection in hardware_monitor.py dashboard
        """)

    st.markdown("---")
    st.subheader("🔄 How Port Mirroring Works")
    st.markdown("""
> The managed switch **copies every packet** from the injector port to the monitor port.
> The monitor PC sees all traffic as if it were sitting inline — without disrupting the network.
> This is how real IDS/IPS systems (Snort, Zeek) are deployed in production.
    """)

    st.subheader("💻 Software Already Ready")
    sw = {
        "Script": ["realtime_sniffer.py", "cascade_inference.py", "hardware_monitor.py", "inject_experiment.py"],
        "Purpose": [
            "Captures live packets, aggregates to flows",
            "Two-phase cascade: LightGBM + IsoForest",
            "Terminal dashboard: live counts + alerts",
            "Injection sweep: 0%–100% Neris mix",
        ],
        "Status": ["✅ Ready", "✅ Ready", "✅ Ready", "✅ Ready"],
    }
    st.dataframe(pd.DataFrame(sw), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Injection Demo
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧪 Injection Demo":
    st.title("🧪 Neris Botnet Injection Experiment")
    st.caption("Simulating attack traffic at controlled injection rates — 0% to 100%")

    df = load_results()
    if df.empty:
        st.error("Results CSV not found. Run inject_experiment.py first.")
        st.stop()

    st.markdown("---")

    # Slider
    rate = st.select_slider(
        "🎚️ Select Injection Level",
        options=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        value=50,
        format_func=lambda x: f"{x}% Neris",
    )

    row = df[df["injection_pct"] == rate].iloc[0]

    st.markdown("---")

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Neris Flows",      f"{int(row.n_neris):,}")
    c2.metric("Benign Flows",     f"{int(row.n_benign):,}")
    c3.metric("Detection (TPR)",  f"{row.tpr_pct:.1f}%",
              delta=None if rate == 0 else f"vs 0%: +{row.tpr_pct:.1f}%")
    c4.metric("False Alarm (FPR)", f"{row.fpr_pct:.1f}%")
    c5.metric("Latency",          f"{row.ms_per_flow:.3f} ms/flow")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    # Left: TPR/FPR bar
    with col_a:
        st.subheader("Detection vs False Alarm")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        bars = ax.bar(["Detection\n(TPR)", "False Alarm\n(FPR)"],
                      [row.tpr_pct, row.fpr_pct],
                      color=["#2ca02c", "#d62728"], width=0.4, zorder=3)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2,
                    b.get_height() + 1.5,
                    f"{b.get_height():.1f}%",
                    ha="center", color="white", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100); ax.set_ylabel("Percentage (%)", color="white")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#333333")
        ax.yaxis.label.set_color("white")
        [t.set_color("white") for t in ax.get_xticklabels()]
        ax.set_title(f"@ {rate}% Injection", color="white", fontsize=11)
        ax.grid(axis="y", color="#333333", zorder=0)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Right: Cascade breakdown pie
    with col_b:
        st.subheader("Cascade Decision Breakdown")
        labels  = ["KNOWN Botnet", "NOVEL Behavior", "Benign"]
        values  = [int(row.known_botnet_count),
                   int(row.novel_behavior_count),
                   int(row.benign_count)]
        colors  = ["#d62728", "#ff7f0e", "#2ca02c"]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "white", "fontsize": 8},
        )
        for at in autotexts:
            at.set_color("white"); at.set_fontsize(8)
        ax.set_title(f"10,000 flows @ {rate}% injection", color="white", fontsize=10)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")
    st.subheader("📈 Full Sweep — TPR & FPR Across All Injection Levels")

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
    ax.plot(df["injection_pct"], df["tpr_pct"], "o-",
            color="#2ca02c", linewidth=2.5, markersize=7, label="Detection Rate (TPR %)")
    ax.plot(df["injection_pct"], df["fpr_pct"], "s--",
            color="#d62728", linewidth=2, markersize=6, label="False Alarm Rate (FPR %)")
    ax.axvline(x=rate, color="white", linestyle=":", linewidth=1.5, alpha=0.6,
               label=f"Current: {rate}%")
    ax.set_xlabel("Injection %", color="white")
    ax.set_ylabel("Rate (%)", color="white")
    ax.set_xticks(df["injection_pct"])
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.grid(True, color="#333333", alpha=0.5)
    ax.set_title("Detection Rate vs False Alarm Rate — All Injection Levels",
                 color="white", fontsize=11)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")
    st.subheader("📊 Cascade Breakdown — All Levels")

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
    x = df["injection_pct"].values
    ax.bar(x, df["known_botnet_count"],  color="#d62728", label="KNOWN Botnet",  width=6)
    ax.bar(x, df["novel_behavior_count"],color="#ff7f0e", label="NOVEL Behavior", width=6,
           bottom=df["known_botnet_count"])
    ax.bar(x, df["benign_count"],        color="#2ca02c", label="Benign",          width=6,
           bottom=df["known_botnet_count"] + df["novel_behavior_count"])
    ax.set_xlabel("Injection %", color="white")
    ax.set_ylabel("Flow Count", color="white")
    ax.set_xticks(x)
    ax.tick_params(colors="white"); ax.spines[:].set_color("#333333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.grid(axis="y", color="#333333", alpha=0.4)
    ax.set_title("Stacked Cascade Decisions per Injection Level",
                 color="white", fontsize=11)
    st.pyplot(fig, use_container_width=True); plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Results
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.title("🤖 Model Training & Evaluation Results")

    tab1, tab2, tab3 = st.tabs(
        ["📌 Phase 1 — Binary", "📌 Phase 1 — Multiclass", "📌 Phase 2 — Zero-Day"]
    )

    with tab1:
        st.subheader("Binary Classification — Botnet vs Benign")
        st.caption("Trained on 12.16M flows | Tested on 7.74M flows")

        binary = {
            "Model":        ["LightGBM ⭐", "Random Forest", "XGBoost", "MLP", "Logistic Reg", "Linear SVM"],
            "Accuracy":     ["91.07%", "91.97%", "90.80%", "89.00%", "~85%", "~84%"],
            "ROC-AUC":      ["0.969", "0.978", "0.964", "—", "—", "—"],
            "Precision":    ["0.91", "0.92", "0.91", "0.90", "—", "—"],
            "Recall":       ["0.91", "0.92", "0.91", "0.89", "—", "—"],
            "Train Time":   ["3.8s ✅", "245s", "18s", "~60s", "fast", "fast"],
            "Deployed":     ["✅ YES", "❌", "❌", "❌", "❌", "❌"],
        }
        st.dataframe(pd.DataFrame(binary), use_container_width=True, hide_index=True)

        st.info(
            "**Why LightGBM?** Best speed-accuracy ratio. "
            "Trains 64× faster than Random Forest with only 0.9% accuracy difference. "
            "Critical for real-time deployment."
        )

        if (FIG_DIR / "phase1_binary_comparison.png").exists():
            st.image(str(FIG_DIR / "phase1_binary_comparison.png"),
                     caption="Binary Model Comparison", use_container_width=True)
        if (FIG_DIR / "phase1_binary_timing.png").exists():
            st.image(str(FIG_DIR / "phase1_binary_timing.png"),
                     caption="Training Time Comparison", use_container_width=True)

    with tab2:
        st.subheader("Multiclass Family Identification — 7 Botnet Families")
        st.caption("Identifies which family: Neris, Rbot, Virut, Murlo, Menti, Sogou, or Unknown")

        multi = {
            "Model":        ["LightGBM ⭐", "Random Forest", "XGBoost", "MLP", "Logistic Reg", "Linear SVM"],
            "Accuracy":     ["71.96%", "69.80%", "70.20%", "66.50%", "~58%", "~55%"],
            "Weighted F1":  ["74.40%", "72.10%", "73.00%", "68.90%", "—", "—"],
            "Train Time":   ["4.1s ✅", "312s", "22s", "~90s", "fast", "fast"],
            "Deployed":     ["✅ YES", "❌", "❌", "❌", "❌", "❌"],
        }
        st.dataframe(pd.DataFrame(multi), use_container_width=True, hide_index=True)

        st.warning(
            "**Why 72%?** 7-class problem with severe imbalance — "
            "Sogou has only 63 flows vs Neris's 246,889. "
            "Neris & Rbot achieve >90% per-class F1. Overall 72% is strong for this setup."
        )

        if (FIG_DIR / "phase1_multiclass_comparison.png").exists():
            st.image(str(FIG_DIR / "phase1_multiclass_comparison.png"),
                     caption="Multiclass Model Comparison", use_container_width=True)

    with tab3:
        st.subheader("Zero-Day Anomaly Detection — NsisAy (Never Seen in Training)")
        st.caption("NsisAy holdout: 2,168 flows | Models trained on benign ONLY")

        st.error(
            "🔒 **NsisAy was completely hidden during all training phases.** \n\n"
            "The anomaly detectors were trained only on benign traffic. "
            "NsisAy flows are presented as 'unknown' — can the system flag them?"
        )

        anomaly = {
            "Model":             ["LOF ⭐", "Isolation Forest", "One-Class SVM", "Elliptic Envelope"],
            "TPR (NsisAy caught)":["35.89%", "28.40%", "22.10%", "18.70%"],
            "FPR (Benign flagged)":["5.0%",  "5.0%",   "5.0%",   "5.0%"],
            "Train Time":         ["28s",    "12s",     "45s",    "8s"],
            "Deployed":           ["✅ YES", "✅ Also",  "❌",     "❌"],
        }
        st.dataframe(pd.DataFrame(anomaly), use_container_width=True, hide_index=True)

        st.success(
            "**35.89% of a completely unseen botnet family** was flagged as anomalous "
            "with only 5% false alarm rate — without any labelled examples of NsisAy."
        )

        if (FIG_DIR / "phase2_anomaly_comparison.png").exists():
            st.image(str(FIG_DIR / "phase2_anomaly_comparison.png"),
                     caption="Anomaly Model Comparison", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — Charts
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Charts":
    st.title("📊 Generated Charts")

    chart_groups = [
        ("🧪 Injection Experiment", [
            (INJ_DIR / "injection_tpr_fpr.png",         "TPR & FPR vs Injection Level"),
            (INJ_DIR / "injection_cascade_breakdown.png","Cascade Decision Breakdown"),
            (INJ_DIR / "injection_latency.png",          "Inference Latency per Level"),
        ]),
        ("📌 Phase 1 — Binary", [
            (FIG_DIR / "phase1_binary_comparison.png",  "Accuracy Comparison"),
            (FIG_DIR / "phase1_binary_timing.png",       "Training Time Comparison"),
        ]),
        ("📌 Phase 1 — Multiclass", [
            (FIG_DIR / "phase1_multiclass_comparison.png", "Accuracy Comparison"),
            (FIG_DIR / "phase1_multiclass_timing.png",      "Training Time Comparison"),
        ]),
        ("📌 Phase 2 — Anomaly", [
            (FIG_DIR / "phase2_anomaly_comparison.png", "Anomaly Model Comparison"),
            (FIG_DIR / "phase2_anomaly_timing.png",      "Training Time Comparison"),
        ]),
    ]

    for group_label, charts in chart_groups:
        st.subheader(group_label)
        cols = st.columns(min(len(charts), 3))
        for col, (path, caption) in zip(cols, charts):
            if path.exists():
                col.image(str(path), caption=caption, use_container_width=True)
            else:
                col.warning(f"Not found: {path.name}")
        st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — Full Report
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📋 Full Report":
    st.title("📋 Full Injection Results Report")

    df = load_results()
    if df.empty:
        st.error("Results CSV not found. Run inject_experiment.py first.")
        st.stop()

    st.subheader("Complete Results Table — Neris Injection Sweep (0% → 100%)")

    display = df[[
        "injection_pct", "n_neris", "n_benign",
        "tpr_pct", "fpr_pct",
        "known_botnet_count", "novel_behavior_count", "benign_count",
        "ms_per_flow",
    ]].rename(columns={
        "injection_pct":         "Injection %",
        "n_neris":               "Neris Flows",
        "n_benign":              "Benign Flows",
        "tpr_pct":               "Detection TPR %",
        "fpr_pct":               "False Alarm FPR %",
        "known_botnet_count":    "KNOWN Botnet",
        "novel_behavior_count":  "NOVEL Behavior",
        "benign_count":          "Benign",
        "ms_per_flow":           "Latency ms/flow",
    })

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔍 Key Findings")

    st.markdown("""
| Finding | Detail |
|---|---|
| **Immediate detection at 10%** | 84.6% TPR even when only 1 in 10 flows is malicious |
| **Stable TPR across all rates** | ~85% from 10% to 100% — model is injection-rate-agnostic |
| **Consistent FPR** | 7.4–7.8% false alarm across mixed-traffic conditions |
| **Zero-day capability** | 35.89% of unseen NsisAy flows surfaced as NOVEL_BEHAVIOR |
| **Real-time speed** | 0.044 ms/flow = 22,700 flows/second on laptop hardware |
| **Hardware ready** | realtime_sniffer.py + cascade_inference.py ready for live capture |
    """)

    st.markdown("---")
    st.subheader("✅ Project Checklist")

    checklist = {
        "Task": [
            "CTU-13 data extraction + cleaning (19.97M flows)",
            "Zero-day holdout split (NsisAy isolated)",
            "Phase 0: Family feature analysis",
            "Phase 1: Binary classification (6 models) — LightGBM 91%",
            "Phase 1: Multiclass family ID (7 families) — LightGBM 72%",
            "Phase 2: Anomaly detection (4 models) — LOF 35.89% TPR",
            "Cascade inference engine (25 unit tests pass)",
            "inject_experiment.py — 11-level sweep 0%→100%",
            "realtime_sniffer.py — live packet capture",
            "hardware_monitor.py — terminal dashboard",
            "Streamlit dashboard (this app)",
            "Physical hardware testbed (switch + adapters)",
        ],
        "Status": [
            "✅ Done", "✅ Done", "✅ Done", "✅ Done",
            "✅ Done", "✅ Done", "✅ Done", "✅ Done",
            "✅ Done", "✅ Done", "✅ Done", "🔲 Pending",
        ],
    }
    st.dataframe(pd.DataFrame(checklist), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📌 Summary Statement")
    st.success(
        "The BotnetLens system demonstrates:\n\n"
        "- **91% binary detection accuracy** on 19.97M real CTU-13 flows\n"
        "- **72% multiclass family identification** across 7 botnet families\n"
        "- **35.89% zero-day detection** on a completely unseen family (NsisAy)\n"
        "- **~85% detection rate** across ALL injection levels (0%–100%) in testbed\n"
        "- **0.044 ms/flow inference** — ready for real-time hardware deployment"
    )
