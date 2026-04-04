from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_risk_predictor.data import (  # noqa: E402
    COMPLEXITY_LEVELS,
    CONTRACT_TYPES,
    PROJECT_TYPES,
    REGIONS,
    RISK_LEVELS,
    default_project_input,
    generate_synthetic_projects,
)
from project_risk_predictor.modeling import predict_project, train_models  # noqa: E402


st.set_page_config(
    page_title="Project Risk Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


PALETTE = {
    "bg": "#f6f1e8",
    "card": "#fbf7f2",
    "ink": "#1f2a1f",
    "muted": "#5d665d",
    "accent": "#b85c38",
    "accent_soft": "#f1d8c7",
    "secondary": "#7a4633",
    "border": "#d9cbb8",
    "danger": "#8e3b2e",
}


@st.cache_data(show_spinner=False)
def load_dataset(sample_size: int) -> pd.DataFrame:
    return generate_synthetic_projects(n_samples=sample_size, seed=42)


@st.cache_resource(show_spinner=False)
def load_training_bundle(sample_size: int):
    dataset = load_dataset(sample_size)
    return train_models(dataset)


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top right, #ead6c6 0%, rgba(234,214,198,0.18) 26%, transparent 27%),
                linear-gradient(180deg, {PALETTE["bg"]} 0%, #efe6da 100%);
            color: {PALETTE["ink"]};
        }}
        .block-container {{
            padding-top: 1.45rem;
            padding-bottom: 1.5rem;
            max-width: 1260px;
        }}
        h1, h2, h3 {{
            color: {PALETTE["ink"]};
            letter-spacing: -0.02em;
            font-weight: 700;
        }}
        p, li, label, .stMarkdown, .stCaption {{
            color: {PALETTE["muted"]};
        }}
        div[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #43251d 0%, #2d1813 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }}
        div[data-testid="stSidebar"] * {{
            color: #eef4ee !important;
        }}
        .hero-card {{
            background: linear-gradient(135deg, rgba(184,92,56,0.96) 0%, rgba(122,70,51,0.96) 100%);
            color: white;
            padding: 1.35rem 1.5rem;
            border-radius: 18px;
            box-shadow: 0 16px 34px rgba(53, 50, 45, 0.10);
            border: 1px solid rgba(255,255,255,0.18);
            margin-bottom: 0.8rem;
        }}
        .hero-card h1, .hero-card p {{
            color: white !important;
            margin: 0;
        }}
        .hero-kicker {{
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.75rem;
            opacity: 0.82;
            margin-bottom: 0.5rem;
        }}
        .surface {{
            background: rgba(251, 247, 242, 0.88);
            border: 1px solid {PALETTE["border"]};
            border-radius: 14px;
            padding: 1rem 1rem;
            box-shadow: 0 8px 20px rgba(47, 44, 39, 0.05);
        }}
        .chart-surface {{
            background: linear-gradient(180deg, rgba(246, 236, 223, 0.94) 0%, rgba(251, 247, 242, 0.90) 100%);
            border: 1px solid rgba(184, 92, 56, 0.14);
            border-radius: 16px;
            padding: 1rem 1rem 0.55rem;
            box-shadow: 0 10px 22px rgba(47, 44, 39, 0.05);
        }}
        .surface-tight {{
            background: rgba(251, 247, 242, 0.72);
            border: 1px solid {PALETTE["border"]};
            border-radius: 12px;
            padding: 0.9rem 0.95rem;
            min-height: 132px;
        }}
        .mini-label {{
            color: {PALETTE["muted"]};
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.28rem;
        }}
        .readout-value {{
            font-size: 1.15rem;
            line-height: 1.15;
            font-weight: 700;
            color: {PALETTE["ink"]};
            margin: 0.15rem 0 0.35rem;
        }}
        .readout-copy {{
            font-size: 0.84rem;
            color: {PALETTE["muted"]};
            margin-bottom: 0.55rem;
        }}
        .metric-value {{
            font-size: 1.75rem;
            line-height: 1;
            font-weight: 700;
            color: {PALETTE["ink"]};
            margin-bottom: 0.22rem;
        }}
        .metric-note {{
            font-size: 0.88rem;
            color: {PALETTE["muted"]};
        }}
        .risk-pill {{
            display: inline-block;
            padding: 0.24rem 0.6rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            background: {PALETTE["accent_soft"]};
            color: {PALETTE["danger"]};
            margin-bottom: 0.6rem;
        }}
        .section-title {{
            font-size: 1rem;
            font-weight: 700;
            color: {PALETTE["ink"]};
            margin-bottom: 0.12rem;
        }}
        .section-copy {{
            font-size: 0.89rem;
            color: {PALETTE["muted"]};
            margin-bottom: 0.65rem;
        }}
        .chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.2rem;
        }}
        .chip {{
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.58rem;
            border-radius: 999px;
            background: rgba(251, 247, 242, 0.72);
            border: 1px solid {PALETTE["border"]};
            color: {PALETTE["ink"]};
            font-size: 0.76rem;
        }}
        .insight-list {{
            margin: 0;
            padding-left: 1.05rem;
        }}
        .insight-list li {{
            margin-bottom: 0.45rem;
            color: {PALETTE["ink"]};
        }}
        .signal-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.6rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.4rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(251, 247, 242, 0.74);
            border-radius: 10px;
            padding: 0.38rem 0.9rem;
            border: 1px solid {PALETTE["border"]};
            font-size: 0.92rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: {PALETTE["secondary"]} !important;
            color: #fdf7f1 !important;
            border-color: {PALETTE["secondary"]} !important;
        }}
        .stTabs [aria-selected="true"] p {{
            color: #fdf7f1 !important;
        }}
        div[data-testid="stMetric"] {{
            background: rgba(251,247,242,0.85);
            border: 1px solid {PALETTE["border"]};
            padding: 0.8rem 0.9rem;
            border-radius: 12px;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
        }}
        div[data-testid="stVegaLiteChart"] {{
            background: transparent !important;
        }}
        .stSlider, .stSelectbox, .stSelectSlider {{
            margin-bottom: 0.1rem;
        }}
        div[data-testid="stProgressBar"] > div > div {{
            background: linear-gradient(90deg, {PALETTE["accent"]} 0%, {PALETTE["secondary"]} 100%);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(bundle) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">Project Intelligence Dashboard</div>
            <h1>Project Risk Predictor</h1>
            <p>Evaluate schedule pressure and budget exposure using project inputs, operational conditions, and side-by-side planning views.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    metric_cards = [
        ("Delay score reliability", f"{bundle.metrics['delay_auc']:.3f}", "how well risk levels separate"),
        ("Budget score reliability", f"{bundle.metrics['budget_auc']:.3f}", "how clearly cost pressure is identified"),
        ("Average delay gap", f"{bundle.metrics['delay_mae']:.1f} days", "typical difference from forecast"),
        ("Project records", f"{bundle.metrics['records']:,}", "records used for scoring"),
    ]
    for col, (label, value, note) in zip((c1, c2, c3, c4), metric_cards):
        with col:
            st.markdown(
                f"""
                <div class="surface-tight">
                    <div class="mini-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def risk_band(probability: float) -> str:
    if probability >= 0.75:
        return "High risk"
    if probability >= 0.45:
        return "Moderate risk"
    return "Low risk"


def risk_tone(probability: float) -> str:
    if probability >= 0.75:
        return "Immediate attention"
    if probability >= 0.45:
        return "Watch closely"
    return "Within plan"


def pressure_value(level: str, low: int = 20, medium: int = 55, high: int = 90) -> int:
    return {"Low": low, "Medium": medium, "High": high}[level]


def build_priority_actions(project_input: dict[str, object], prediction: dict[str, float]) -> list[str]:
    actions: list[str] = []
    if project_input["material_risk"] == "High":
        actions.append("Lock procurement milestones early and confirm backup suppliers for exposed materials.")
    if int(project_input["permit_delay_days"]) >= 20:
        actions.append("Escalate permitting follow-up and separate approval-critical work from non-dependent activities.")
    if int(project_input["change_order_count"]) >= 5:
        actions.append("Tighten change management so field updates stop eroding crew coordination.")
    if int(project_input["crew_size"]) <= 20 and prediction["delay_probability"] >= 0.45:
        actions.append("Rebalance labor coverage on critical path activities before the next planning cycle.")
    if float(project_input["equipment_utilization"]) <= 60:
        actions.append("Review equipment idle time and sequence conflicts to recover site productivity.")
    if not actions:
        actions.append("Current setup is relatively stable. Maintain permit follow-up, trade coordination, and procurement visibility.")
    return actions[:3]


def style_chart(chart: alt.Chart | alt.LayerChart) -> alt.Chart | alt.LayerChart:
    return chart.configure_view(stroke=None).configure_axis(
        domain=False,
        gridColor=PALETTE["border"],
        gridOpacity=0.28,
        labelColor=PALETTE["muted"],
        labelFontSize=11,
        tickColor=PALETTE["border"],
        title=None,
    ).configure_legend(
        labelColor=PALETTE["muted"],
        titleColor=PALETTE["ink"],
        labelFontSize=11,
        symbolType="circle",
    ).configure(background="transparent")


def chart_panel(title: str, copy: str) -> None:
    st.markdown('<div class="chart-surface">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def sparkline_svg(values: list[float], color: str) -> str:
    width = 180
    height = 48
    if not values:
        values = [0.0, 0.0]
    minimum = min(values)
    maximum = max(values)
    spread = maximum - minimum or 1.0
    points = []
    for idx, value in enumerate(values):
        x = idx * (width / max(len(values) - 1, 1))
        y = height - (((value - minimum) / spread) * (height - 10) + 5)
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    area = f"0,{height} " + polyline + f" {width},{height}"
    return f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none">
        <polygon points="{area}" fill="{color}" opacity="0.14"></polygon>
        <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></polyline>
    </svg>
    """


def build_project_form(defaults: dict[str, object], key_prefix: str) -> dict[str, object]:
    top1, top2, top3 = st.columns(3)
    with top1:
        project_type = st.selectbox("Project type", PROJECT_TYPES, index=PROJECT_TYPES.index(defaults["project_type"]), key=f"{key_prefix}_project_type")
    with top2:
        region = st.selectbox("Region", REGIONS, index=REGIONS.index(defaults["region"]), key=f"{key_prefix}_region")
    with top3:
        contract_type = st.selectbox("Contract type", CONTRACT_TYPES, index=CONTRACT_TYPES.index(defaults["contract_type"]), key=f"{key_prefix}_contract_type")

    col1, col2, col3 = st.columns(3)
    with col1:
        budget_musd = st.slider("Budget (USD millions)", 2.0, 160.0, float(defaults["budget_musd"]), step=0.5, key=f"{key_prefix}_budget")
        planned_duration_days = st.slider("Planned duration (days)", 60, 950, int(defaults["planned_duration_days"]), step=5, key=f"{key_prefix}_duration")
        permit_delay_days = st.slider("Permit delay days", 0, 90, int(defaults["permit_delay_days"]), key=f"{key_prefix}_permit_delay")
        client_decision_latency = st.slider("Client decision latency (days)", 1, 25, int(defaults["client_decision_latency"]), key=f"{key_prefix}_client_latency")
        equipment_utilization = st.slider("Equipment utilization (%)", 35.0, 98.0, float(defaults["equipment_utilization"]), step=1.0, key=f"{key_prefix}_equipment")
    with col2:
        crew_size = st.slider("Crew size", 6, 140, int(defaults["crew_size"]), key=f"{key_prefix}_crew")
        subcontractor_count = st.slider("Subcontractors", 1, 25, int(defaults["subcontractor_count"]), key=f"{key_prefix}_subs")
        change_order_count = st.slider("Change orders", 0, 18, int(defaults["change_order_count"]), key=f"{key_prefix}_change_orders")
        safety_incidents = st.slider("Safety incidents to date", 0, 8, int(defaults["safety_incidents"]), key=f"{key_prefix}_incidents")
        percent_self_performed = st.slider("Self-performed work (%)", 5.0, 95.0, float(defaults["percent_self_performed"]), step=1.0, key=f"{key_prefix}_self_performed")
    with col3:
        weather_severity = st.select_slider("Weather severity", options=RISK_LEVELS, value=defaults["weather_severity"], key=f"{key_prefix}_weather")
        material_risk = st.select_slider("Material delivery risk", options=RISK_LEVELS, value=defaults["material_risk"], key=f"{key_prefix}_material")
        labor_availability = st.select_slider("Labor shortage risk", options=RISK_LEVELS, value=defaults["labor_availability"], key=f"{key_prefix}_labor")
        site_complexity = st.select_slider("Site complexity", options=COMPLEXITY_LEVELS, value=defaults["site_complexity"], key=f"{key_prefix}_complexity")
        site_density = st.select_slider("Site density", options=COMPLEXITY_LEVELS, value=defaults["site_density"], key=f"{key_prefix}_density")

    return {
        "project_type": project_type,
        "region": region,
        "contract_type": contract_type,
        "budget_musd": budget_musd,
        "planned_duration_days": planned_duration_days,
        "crew_size": crew_size,
        "subcontractor_count": subcontractor_count,
        "change_order_count": change_order_count,
        "safety_incidents": safety_incidents,
        "permit_delay_days": permit_delay_days,
        "client_decision_latency": client_decision_latency,
        "weather_severity": weather_severity,
        "material_risk": material_risk,
        "labor_availability": labor_availability,
        "site_complexity": site_complexity,
        "site_density": site_density,
        "percent_self_performed": percent_self_performed,
        "equipment_utilization": equipment_utilization,
    }


def render_prediction_summary(prediction: dict[str, float]) -> None:
    delay_band = risk_band(prediction["delay_probability"])
    budget_band = risk_band(prediction["budget_probability"])
    cols = st.columns(3)
    cards = [
        (
            "Delay risk",
            f"{prediction['delay_probability']:.1%}",
            delay_band,
            [18, 26, 41, prediction["delay_probability"] * 100],
            PALETTE["accent"],
        ),
        (
            "Budget overrun risk",
            f"{prediction['budget_probability']:.1%}",
            budget_band,
            [14, 22, 31, prediction["budget_probability"] * 100],
            PALETTE["secondary"],
        ),
        (
            "Expected delay",
            f"{prediction['expected_delay_days']:.0f} days",
            "forecasted slip",
            [
                max(prediction["expected_delay_days"] * 0.25, 2),
                max(prediction["expected_delay_days"] * 0.45, 4),
                max(prediction["expected_delay_days"] * 0.7, 7),
                prediction["expected_delay_days"],
            ],
            PALETTE["danger"],
        ),
    ]
    for col, (label, value, note, series, color) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="surface-tight">
                    <div class="mini-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                    <div style="margin-top:0.55rem;">{sparkline_svg(series, color)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_context_chips(project_input: dict[str, object]) -> None:
    chips = [
        f"Type: {project_input['project_type']}",
        f"Region: {project_input['region']}",
        f"Contract: {project_input['contract_type']}",
        f"Budget: ${float(project_input['budget_musd']):.1f}M",
        f"Duration: {int(project_input['planned_duration_days'])} days",
        f"Crew: {int(project_input['crew_size'])}",
    ]
    markup = "".join(f"<span class='chip'>{chip}</span>" for chip in chips)
    st.markdown(f"<div class='chip-row'>{markup}</div>", unsafe_allow_html=True)


def render_snapshot_overview(project_input: dict[str, object], prediction: dict[str, float]) -> None:
    left, right = st.columns([1.15, 0.85])
    actions = build_priority_actions(project_input, prediction)

    with left:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="risk-pill">{risk_tone(prediction["delay_probability"])}</div>
            <div class="section-title">Project readout</div>
            <div class="readout-value">{prediction['delay_probability']:.0%} delay risk, {prediction['budget_probability']:.0%} budget risk</div>
            <div class="readout-copy">Expected schedule slip: {prediction['expected_delay_days']:.0f} days.</div>
            """,
            unsafe_allow_html=True,
        )
        render_context_chips(project_input)
        a, b, c = st.columns(3)
        a.markdown(f"**Schedule outlook**  \n{risk_tone(prediction['delay_probability'])}")
        b.markdown(f"**Budget outlook**  \n{risk_tone(prediction['budget_probability'])}")
        c.markdown(f"**Project type**  \n{project_input['project_type']}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-title">Recommended actions</div>
            <div class="section-copy">Three practical moves to improve delivery confidence.</div>
            """,
            unsafe_allow_html=True,
        )
        for action in actions:
            st.write(f"- {action}")
        st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_charts(project_input: dict[str, object], prediction: dict[str, float]) -> None:
    risk_frame = pd.DataFrame(
        {
            "measure": ["Delay risk", "Budget risk", "Schedule pressure"],
            "value": [
                prediction["delay_probability"] * 100,
                prediction["budget_probability"] * 100,
                min(prediction["expected_delay_days"] / 60 * 100, 100),
            ],
            "color": [PALETTE["accent"], PALETTE["secondary"], PALETTE["danger"]],
        }
    )
    pressure_frame = pd.DataFrame(
        {
            "factor": [
                "Material risk",
                "Labor pressure",
                "Site complexity",
                "Site density",
                "Permit delays",
                "Change orders",
            ],
            "value": [
                pressure_value(str(project_input["material_risk"])),
                pressure_value(str(project_input["labor_availability"])),
                pressure_value(str(project_input["site_complexity"]), 25, 55, 88),
                pressure_value(str(project_input["site_density"]), 25, 55, 88),
                min(float(project_input["permit_delay_days"]) / 45 * 100, 100),
                min(float(project_input["change_order_count"]) / 10 * 100, 100),
            ],
        }
    )
    timeline_frame = pd.DataFrame(
        {
            "stage": ["Permitting", "Mobilization", "Core work", "Risk buffer"],
            "days": [
                min(float(project_input["permit_delay_days"]), 45),
                min(float(project_input["client_decision_latency"]) * 2.5, 45),
                min(float(project_input["planned_duration_days"]) * 0.55, 180),
                min(float(prediction["expected_delay_days"]), 60),
            ],
            "phase": ["Front-end", "Front-end", "Execution", "Buffer"],
        }
    )

    risk_chart = (
        alt.Chart(risk_frame)
        .mark_bar(cornerRadiusEnd=6, size=28)
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 100]), title=None),
            y=alt.Y("measure:N", sort=None, title=None),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[alt.Tooltip("measure:N", title="Metric"), alt.Tooltip("value:Q", title="Value", format=".1f")],
        )
        .properties(height=180)
    )
    risk_text = risk_chart.mark_text(align="left", baseline="middle", dx=8, color=PALETTE["ink"]).encode(
        text=alt.Text("value:Q", format=".0f")
    )

    pressure_chart = (
        alt.Chart(pressure_frame)
        .mark_bar(cornerRadiusEnd=6, size=20, color=PALETTE["accent_soft"])
        .encode(
            y=alt.Y("factor:N", sort="-x", title=None),
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 100]), title=None),
            tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("value:Q", title="Intensity", format=".0f")],
        )
        .properties(height=220)
    )
    pressure_points = (
        alt.Chart(pressure_frame)
        .mark_point(size=95, filled=True, color=PALETTE["accent"])
        .encode(
            y=alt.Y("factor:N", sort="-x", title=None),
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 100]), title=None),
        )
    )
    pressure_text = (
        alt.Chart(pressure_frame)
        .mark_text(align="left", baseline="middle", dx=8, color=PALETTE["ink"])
        .encode(
            y=alt.Y("factor:N", sort="-x", title=None),
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 100]), title=None),
            text=alt.Text("value:Q", format=".0f"),
        )
    )
    timeline_chart = (
        alt.Chart(timeline_frame)
        .mark_bar(cornerRadius=5)
        .encode(
            x=alt.X("days:Q", title="Days"),
            y=alt.Y("stage:N", sort=None, title=None),
            color=alt.Color(
                "phase:N",
                scale=alt.Scale(
                    domain=["Front-end", "Execution", "Buffer"],
                    range=[PALETTE["accent_soft"], PALETTE["secondary"], PALETTE["danger"]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("stage:N", title="Stage"),
                alt.Tooltip("days:Q", title="Days", format=".0f"),
                alt.Tooltip("phase:N", title="Category"),
            ],
        )
        .properties(height=180)
    )

    left, right = st.columns(2)
    with left:
        chart_panel("Risk summary", "A quick view of the three main outcome signals for this project.")
        st.altair_chart(style_chart(risk_chart + risk_text), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        chart_panel("Operating pressure profile", "A visual read on the main execution pressures in the current plan.")
        st.altair_chart(style_chart(pressure_chart + pressure_points + pressure_text), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    chart_panel("Schedule shape", "A planning view showing where front-end friction and schedule buffer are accumulating.")
    st.altair_chart(style_chart(timeline_chart), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_signal_chart(feature_importances: pd.DataFrame) -> None:
    label_map = {
        "planned_duration_days": "Planned duration",
        "change_order_count": "Change orders",
        "permit_delay_days": "Permit delays",
        "client_decision_latency": "Client decision speed",
        "equipment_utilization": "Equipment utilization",
        "crew_size": "Crew size",
        "subcontractor_count": "Subcontractors",
        "safety_incidents": "Safety incidents",
        "budget_musd": "Budget",
        "percent_self_performed": "Self-performed work",
        "material_risk_High": "High material risk",
        "material_risk_Medium": "Medium material risk",
        "labor_availability_High": "High labor pressure",
        "labor_availability_Medium": "Moderate labor pressure",
        "site_complexity_High": "High site complexity",
        "site_density_High": "Dense site conditions",
        "weather_severity_High": "Severe weather exposure",
    }
    top_signals = feature_importances.head(6).copy()
    top_signals["signal"] = top_signals["feature"].map(lambda value: label_map.get(value, str(value).replace("_", " ").title()))
    chart = (
        alt.Chart(top_signals)
        .mark_bar(cornerRadiusEnd=6, color=PALETTE["secondary"], size=22)
        .encode(
            y=alt.Y("signal:N", sort="-x", title=None),
            x=alt.X("importance:Q", title=None),
            tooltip=[alt.Tooltip("signal:N", title="Signal"), alt.Tooltip("importance:Q", title="Influence", format=".2f")],
        )
        .properties(height=220)
    )
    text = (
        alt.Chart(top_signals)
        .mark_text(align="left", baseline="middle", dx=8, color=PALETTE["ink"])
        .encode(
            y=alt.Y("signal:N", sort="-x", title=None),
            x=alt.X("importance:Q", title=None),
            text=alt.Text("importance:Q", format=".2f"),
        )
    )
    st.altair_chart(style_chart(chart + text), use_container_width=True)


def render_driver_summary(project_input: dict[str, object], feature_importances: pd.DataFrame) -> None:
    driver_labels = []
    if project_input["change_order_count"] >= 5:
        driver_labels.append("Frequent change orders are increasing coordination strain.")
    if project_input["permit_delay_days"] >= 20:
        driver_labels.append("Permitting friction is materially reducing schedule confidence.")
    if project_input["material_risk"] == "High":
        driver_labels.append("Material delivery exposure is pushing both schedule and cost risk upward.")
    if project_input["labor_availability"] == "High":
        driver_labels.append("Labor shortage pressure is lowering execution resilience.")
    if project_input["equipment_utilization"] <= 60:
        driver_labels.append("Low equipment utilization suggests productivity drag on site.")
    if project_input["site_complexity"] == "High":
        driver_labels.append("High site complexity is amplifying uncertainty across trades.")
    if not driver_labels:
        driver_labels.append("This scenario is relatively stable; residual risk is mostly baseline complexity.")

    left, right = st.columns([1.05, 1.2])
    with left:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.markdown('<div class="risk-pill">Key drivers</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-title">Why this project is scoring this way</div>
            <div class="section-copy">A quick summary of the conditions driving the current result.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<ul class='insight-list'>" + "".join(f"<li>{label}</li>" for label in driver_labels[:4]) + "</ul>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        chart_panel("Primary risk signals", "The strongest inputs shaping the current delay outlook. Higher bars mean stronger influence on the delay score.")
        render_signal_chart(feature_importances)
        st.markdown("</div>", unsafe_allow_html=True)


def render_comparison_chart(comparison: pd.DataFrame) -> None:
    long_frame = comparison.melt(id_vars="Scenario", var_name="measure", value_name="value")
    lines = (
        alt.Chart(long_frame)
        .mark_rule(strokeWidth=3, color=PALETTE["border"])
        .encode(
            y=alt.Y("measure:N", sort=None, title=None),
            x=alt.X("value:Q", title=None),
            detail="measure:N",
        )
    )
    points = (
        alt.Chart(long_frame)
        .mark_point(size=120, filled=True)
        .encode(
            y=alt.Y("measure:N", sort=None, title=None),
            x=alt.X("value:Q", title=None),
            color=alt.Color(
                "Scenario:N",
                scale=alt.Scale(domain=list(comparison["Scenario"]), range=[PALETTE["secondary"], PALETTE["accent"]]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("measure:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=".1f"),
            ],
        )
    )
    st.altair_chart(style_chart((lines + points).properties(height=220)), use_container_width=True)


def render_what_if(bundle, base_input: dict[str, object]) -> None:
    st.markdown(
        """
        <div class="section-title">Stress test the plan</div>
        <div class="section-copy">Choose one operational change and compare it against the current plan.</div>
        """,
        unsafe_allow_html=True,
    )
    control = st.selectbox(
        "Planning change",
        [
            "Increase material risk",
            "Add change orders",
            "Reduce crew size",
            "Increase permit delays",
        ],
        key="what_if_selector",
    )
    scenario = dict(base_input)
    if control == "Increase material risk":
        scenario["material_risk"] = "High"
    elif control == "Add change orders":
        scenario["change_order_count"] = min(int(base_input["change_order_count"]) + 4, 18)
    elif control == "Reduce crew size":
        scenario["crew_size"] = max(int(base_input["crew_size"]) - 8, 6)
    else:
        scenario["permit_delay_days"] = min(int(base_input["permit_delay_days"]) + 15, 90)

    base_pred = predict_project(bundle, base_input)
    scenario_pred = predict_project(bundle, scenario)
    comparison = pd.DataFrame(
        [
            {
                "Scenario": "Current plan",
                "Delay risk": round(base_pred["delay_probability"] * 100, 1),
                "Budget risk": round(base_pred["budget_probability"] * 100, 1),
                "Expected delay days": round(base_pred["expected_delay_days"], 1),
            },
            {
                "Scenario": control,
                "Delay risk": round(scenario_pred["delay_probability"] * 100, 1),
                "Budget risk": round(scenario_pred["budget_probability"] * 100, 1),
                "Expected delay days": round(scenario_pred["expected_delay_days"], 1),
            },
        ]
    )

    a, b, c = st.columns(3)
    a.metric("Delay risk delta", f"{comparison.iloc[1]['Delay risk'] - comparison.iloc[0]['Delay risk']:+.1f} pts")
    b.metric("Budget risk delta", f"{comparison.iloc[1]['Budget risk'] - comparison.iloc[0]['Budget risk']:+.1f} pts")
    c.metric("Delay days delta", f"{comparison.iloc[1]['Expected delay days'] - comparison.iloc[0]['Expected delay days']:+.1f}")

    left, right = st.columns([1.2, 1.0])
    with left:
        chart_panel("Plan shift", "A before-and-after comparison of the current plan and the selected operating change.")
        render_comparison_chart(comparison)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="risk-pill">{control}</div>
            <div class="section-title">Planning readout</div>
            <div class="section-copy">Use this view to explain how one operating change shifts project risk.</div>
            <ul class="insight-list">
                <li>Current delay risk: {comparison.iloc[0]["Delay risk"]:.1f}%</li>
                <li>Updated delay risk: {comparison.iloc[1]["Delay risk"]:.1f}%</li>
                <li>Updated budget risk: {comparison.iloc[1]["Budget risk"]:.1f}%</li>
                <li>Expected schedule slip: {comparison.iloc[1]["Expected delay days"]:.1f} days</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_distribution_chart(dataset: pd.DataFrame) -> None:
    chart = (
        alt.Chart(dataset)
        .mark_bar(color=PALETTE["accent"], opacity=0.85)
        .encode(
            x=alt.X("delay_days:Q", bin=alt.Bin(maxbins=24), title="Delay days"),
            y=alt.Y("count():Q", title="Projects"),
            tooltip=[alt.Tooltip("count():Q", title="Projects")],
        )
        .properties(height=220)
    )
    st.altair_chart(style_chart(chart), use_container_width=True)


def render_dataset_explorer(dataset: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="section-title">Project data profile</div>
        <div class="section-copy">A quick look at the project records powering the scoring views.</div>
        """,
        unsafe_allow_html=True,
    )
    summary = pd.DataFrame(
        {
            "Metric": [
                "Projects delayed",
                "Projects over budget",
                "Average delay days",
                "Average cost overrun (%)",
            ],
            "Value": [
                f"{dataset['delayed'].mean():.1%}",
                f"{dataset['over_budget'].mean():.1%}",
                f"{dataset['delay_days'].mean():.1f}",
                f"{dataset['cost_overrun_pct'].mean():.1f}",
            ],
        }
    )
    delayed_by_type = (
        dataset.groupby("project_type", as_index=False)["delayed"]
        .mean()
        .rename(columns={"delayed": "delay_rate"})
        .sort_values("delay_rate", ascending=False)
    )
    budget_by_region = (
        dataset.groupby("region", as_index=False)["over_budget"]
        .mean()
        .rename(columns={"over_budget": "budget_rate"})
        .sort_values("budget_rate", ascending=False)
    )
    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        chart_panel("Project mix", "Quick comparisons across project types and regions.")
        delay_chart = (
            alt.Chart(delayed_by_type)
            .mark_bar(cornerRadiusEnd=6, color=PALETTE["accent"])
            .encode(
                y=alt.Y("project_type:N", sort="-x", title=None),
                x=alt.X("delay_rate:Q", axis=alt.Axis(format="%"), title=None),
                tooltip=[alt.Tooltip("project_type:N", title="Project type"), alt.Tooltip("delay_rate:Q", title="Delay rate", format=".1%")],
            )
            .properties(height=140)
        )
        budget_chart = (
            alt.Chart(budget_by_region)
            .mark_bar(cornerRadiusEnd=6, color=PALETTE["secondary"])
            .encode(
                y=alt.Y("region:N", sort="-x", title=None),
                x=alt.X("budget_rate:Q", axis=alt.Axis(format="%"), title=None),
                tooltip=[alt.Tooltip("region:N", title="Region"), alt.Tooltip("budget_rate:Q", title="Budget risk", format=".1%")],
            )
            .properties(height=140)
        )
        st.altair_chart(style_chart(delay_chart), use_container_width=True)
        st.altair_chart(style_chart(budget_chart), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    chart_panel("Delay distribution", "A quick look at how delay outcomes are spread across the project population.")
    render_distribution_chart(dataset)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="surface">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">Sample records</div>
        <div class="section-copy">A few project rows so the scoring inputs stay visible and inspectable.</div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(dataset.head(20), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar_explainer() -> None:
    with st.sidebar.expander("How this works", expanded=False):
        st.write("This tool scores project risk from structured planning inputs rather than hidden text prompts.")
        st.write("Model family: Random Forest")
        st.write("It uses three machine learning models:")
        st.write("- delay risk classifier")
        st.write("- budget overrun classifier")
        st.write("- expected delay regressor")
        st.write("The scoring engine looks at project scope, crew size, trade coordination, permitting, labor pressure, material exposure, and site conditions.")
        st.write("The charts in `Primary risk signals` show which inputs are most influential in the current scoring logic.")


def main() -> None:
    inject_styles()
    dataset_size = st.sidebar.slider("Project data size", 1000, 8000, 3500, step=500)
    dataset = load_dataset(dataset_size)
    bundle = load_training_bundle(dataset_size)

    st.sidebar.markdown("## Workspace controls")
    st.sidebar.caption("Adjust the project dataset size and review planning outcomes across the dashboard.")
    st.sidebar.metric(
        "Delay score reliability",
        f"{bundle.metrics['delay_auc']:.3f}",
        help="How well the delay model separates lower-risk projects from higher-risk ones. Higher is better.",
    )
    st.sidebar.metric(
        "Budget score reliability",
        f"{bundle.metrics['budget_auc']:.3f}",
        help="How well the budget model distinguishes projects more likely to run over budget from those less likely to do so. Higher is better.",
    )
    st.sidebar.metric(
        "Average delay gap",
        f"{bundle.metrics['delay_mae']:.1f} days",
        help="The average difference between the predicted delay and the actual delay in the evaluation sample. Lower is better.",
    )
    render_sidebar_explainer()

    render_hero(bundle)

    tabs = st.tabs(["Project Snapshot", "Plan Comparison", "Data Explorer"])
    defaults = default_project_input()

    with tabs[0]:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-title">Set up a project</div>
            <div class="section-copy">Enter project details and review schedule pressure and budget exposure.</div>
            """,
            unsafe_allow_html=True,
        )
        project_input = build_project_form(defaults, "single_project")
        st.markdown("</div>", unsafe_allow_html=True)
        prediction = predict_project(bundle, project_input)
        render_prediction_summary(prediction)
        render_snapshot_overview(project_input, prediction)
        render_prediction_charts(project_input, prediction)
        render_driver_summary(project_input, bundle.feature_importances)

    with tabs[1]:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        base_input = build_project_form(defaults, "what_if")
        st.markdown("</div>", unsafe_allow_html=True)
        render_what_if(bundle, base_input)

    with tabs[2]:
        render_dataset_explorer(dataset)


if __name__ == "__main__":
    main()
