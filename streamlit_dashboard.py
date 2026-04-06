import os
import time
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Edge-Cloud CNC Anomaly Dashboard",
    page_icon="Group 2",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.sub-text {
    color: #b0b0b0;
    margin-bottom: 1rem;
}
.status-card {
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    color: white;
    margin-top: 10px;
    margin-bottom: 10px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.healthy {
    background: linear-gradient(90deg, #1faa59, #2ecc71);
}
.warning {
    background: linear-gradient(90deg, #d4ac0d, #f1c40f);
    color: black;
}
.alarm {
    background: linear-gradient(90deg, #c0392b, #e74c3c);
    animation: blink 1s infinite;
}
.masked {
    background: linear-gradient(90deg, #5d6d7e, #95a5a6);
}
.metric-box {
    padding: 12px;
    border-radius: 10px;
    background-color: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}
.block-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
@keyframes blink {
    50% { opacity: 0.45; }
}
</style>
""", unsafe_allow_html=True)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "testing_cycles.npz")

ALARM_PROFILE_PATHS = {
    "Base Profile v1.0": os.path.join(ROOT_DIR, "data", "processed", "cloud_alarm_params.json"),
    "Updated Profile v1.1": os.path.join(ROOT_DIR, "data", "processed", "cloud_alarm_params_new.json"),
    "Tuned Profile v1.2": os.path.join(ROOT_DIR, "data", "processed", "cloud_alarm_params_tuned.json"),
}


@st.cache_data
def load_cycles():
    archive = np.load(TEST_DATA_PATH)
    return {name: archive[name] for name in archive.files}


@st.cache_data
def load_alarm_profile(path):
    with open(path, "r") as f:
        return json.load(f)


def call_cloud_api(api_url, cycle_name, batch_data, machine_state, model_profile_name):
    payload = {
        "cycle_name": cycle_name,
        "machine_state": machine_state,
        "model_version": model_profile_name,
        "data": batch_data.tolist()
    }
    response = requests.post(f"{api_url}/predict", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def compute_status(mae, threshold, masked, violation_counter, debounce_n, recent_maes, subhealth_ratio):
    if masked:
        return "⚪ Masked", violation_counter, False, False

    if mae > threshold:
        violation_counter += 1
    else:
        violation_counter = 0

    alarm = violation_counter >= debounce_n

    if len(recent_maes) > 0:
        exceed_ratio = float(np.mean(np.array(recent_maes) > threshold))
    else:
        exceed_ratio = 0.0

    sub_health = exceed_ratio >= subhealth_ratio and not alarm

    if alarm:
        return "🔴 Anomaly Alarm", violation_counter, alarm, sub_health
    if sub_health:
        return "🟡 Sub-Health Warning", violation_counter, alarm, sub_health
    return "🟢 Healthy", violation_counter, alarm, sub_health


def reset_state():
    st.session_state.records = []
    st.session_state.violation_counter = 0
    st.session_state.current_batch_idx = 0
    st.session_state.running = False


def get_status_css(status_text):
    if "Healthy" in status_text:
        return "healthy"
    if "Warning" in status_text:
        return "warning"
    if "Masked" in status_text:
        return "masked"
    return "alarm"


if "records" not in st.session_state:
    st.session_state.records = []
if "violation_counter" not in st.session_state:
    st.session_state.violation_counter = 0
if "current_batch_idx" not in st.session_state:
    st.session_state.current_batch_idx = 0
if "running" not in st.session_state:
    st.session_state.running = False

cycles = load_cycles()

st.markdown("<div class='main-title'>Group 2 - Edge-Cloud CNC Anomaly Detection Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Live dashboard.</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Controls")

    api_url = st.text_input("Cloud API URL", value="http://127.0.0.1:5000")

    cycle_name = st.selectbox(
        "Select cycle",
        options=list(cycles.keys()),
        index=0
    )

    machine_state = st.selectbox(
        "Machine state",
        options=["RUNNING", "STARTING", "STOPPING", "IDLE"],
        index=0
    )

    profile_name = st.selectbox(
        "Threshold profile",
        options=list(ALARM_PROFILE_PATHS.keys()),
        index=0
    )

    profile = load_alarm_profile(ALARM_PROFILE_PATHS[profile_name])
    mu = float(profile["mu"])
    sigma = float(profile["sigma"])

    st.markdown("### Threshold Settings")
    k = st.slider("Threshold multiplier (k)", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
    threshold = mu + k * sigma

    st.markdown("### Debounce Settings")
    debounce_n = st.number_input("Debounce limit (N)", min_value=1, max_value=20, value=3, step=1)

    st.markdown("### Sub-Health Settings")
    subhealth_window = st.number_input("Sub-health window size", min_value=5, max_value=100, value=20, step=1)
    subhealth_ratio = st.slider("Sub-health trigger ratio", min_value=0.05, max_value=1.0, value=0.20, step=0.05)

    st.markdown("### Streaming Settings")
    batch_size = st.number_input("Batch size", min_value=1, max_value=20, value=5, step=1)
    delay_sec = st.slider("Delay between batches (sec)", min_value=0.0, max_value=2.0, value=0.2, step=0.1)

    st.info(
        f"Selected profile: **{profile.get('version', 'unknown')}**  \n"
        f"μ = **{mu:.6f}**  \n"
        f"σ = **{sigma:.6f}**  \n"
        f"Threshold = **{threshold:.6f}**"
    )

col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("Cycle Information")
    selected_cycle = cycles[cycle_name]
    st.write(f"Shape: `{selected_cycle.shape}`")
    st.write(f"Cycle type: {'Bad' if 'bad' in cycle_name.lower() else 'Good'}")

with col_b:
    st.subheader("Session Actions")
    c1, c2, c3 = st.columns(3)
    if c1.button("Reset"):
        reset_state()
        st.rerun()

    if c2.button("Next Batch"):
        st.session_state.running = False

        n_windows = selected_cycle.shape[0]
        start = st.session_state.current_batch_idx
        end = min(start + batch_size, n_windows)

        if start < n_windows:
            batch = selected_cycle[start:end]
            try:
                result = call_cloud_api(api_url, cycle_name, batch, machine_state, profile_name)
                mae = float(result["mean_mae"])
                masked = machine_state in ["STARTING", "STOPPING"]

                recent_maes = [r["mean_mae"] for r in st.session_state.records[-(subhealth_window - 1):]]
                recent_maes.append(mae)

                status, vc, alarm, sub_health = compute_status(
                    mae=mae,
                    threshold=threshold,
                    masked=masked,
                    violation_counter=st.session_state.violation_counter,
                    debounce_n=debounce_n,
                    recent_maes=recent_maes,
                    subhealth_ratio=subhealth_ratio
                )

                st.session_state.violation_counter = vc
                st.session_state.current_batch_idx = end

                st.session_state.records.append({
                    "batch_start": start,
                    "batch_end": end,
                    "mean_mae": mae,
                    "threshold": threshold,
                    "masked": masked,
                    "status": status,
                    "alarm": alarm,
                    "sub_health": sub_health,
                    "violation_counter": vc,
                    "pct_above_3sigma": result.get("pct_windows_above_3sigma", 0.0),
                    "pct_above_4sigma": result.get("pct_windows_above_4sigma", 0.0)
                })

                st.rerun()

            except Exception as e:
                st.error(f"API call failed: {e}")

    if c3.button("Run Full Cycle"):
        st.session_state.running = True

progress_placeholder = st.empty()
progress_text_placeholder = st.empty()
status_placeholder = st.empty()
metrics_placeholder = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()

n_windows = selected_cycle.shape[0]

if st.session_state.running:
    while st.session_state.current_batch_idx < n_windows:
        start = st.session_state.current_batch_idx
        end = min(start + batch_size, n_windows)
        batch = selected_cycle[start:end]

        try:
            result = call_cloud_api(api_url, cycle_name, batch, machine_state, profile_name)
            mae = float(result["mean_mae"])
            masked = machine_state in ["STARTING", "STOPPING"]

            recent_maes = [r["mean_mae"] for r in st.session_state.records[-(subhealth_window - 1):]]
            recent_maes.append(mae)

            status, vc, alarm, sub_health = compute_status(
                mae=mae,
                threshold=threshold,
                masked=masked,
                violation_counter=st.session_state.violation_counter,
                debounce_n=debounce_n,
                recent_maes=recent_maes,
                subhealth_ratio=subhealth_ratio
            )

            st.session_state.violation_counter = vc
            st.session_state.current_batch_idx = end

            st.session_state.records.append({
                "batch_start": start,
                "batch_end": end,
                "mean_mae": mae,
                "threshold": threshold,
                "masked": masked,
                "status": status,
                "alarm": alarm,
                "sub_health": sub_health,
                "violation_counter": vc,
                "pct_above_3sigma": result.get("pct_windows_above_3sigma", 0.0),
                "pct_above_4sigma": result.get("pct_windows_above_4sigma", 0.0)
            })

            df = pd.DataFrame(st.session_state.records)
            progress_placeholder.progress(min(end / n_windows, 1.0))
            progress_text_placeholder.caption(f"Progress: {end}/{n_windows} windows processed")

            latest = df.iloc[-1]
            css_class = get_status_css(latest["status"])
            status_placeholder.markdown(
                f"<div class='status-card {css_class}'>Status: {latest['status']}</div>",
                unsafe_allow_html=True
            )

            m1, m2, m3, m4 = metrics_placeholder.columns(4)
            m1.metric("Latest MAE", f"{latest['mean_mae']:.6f}")
            m2.metric("Threshold", f"{latest['threshold']:.6f}")
            m3.metric("Violations", int(latest["violation_counter"]))
            m4.metric("Current Status", latest["status"])

            chart_df = df[["mean_mae", "threshold"]].copy()
            chart_placeholder.line_chart(chart_df)

            table_placeholder.dataframe(
                df.tail(20),
                use_container_width=True,
                hide_index=True
            )

            time.sleep(delay_sec)

        except Exception as e:
            st.error(f"Streaming stopped due to API error: {e}")
            st.session_state.running = False
            break

    st.session_state.running = False

if len(st.session_state.records) > 0:
    df = pd.DataFrame(st.session_state.records)

    progress_placeholder.progress(min(st.session_state.current_batch_idx / n_windows, 1.0))
    progress_text_placeholder.caption(
        f"Progress: {st.session_state.current_batch_idx}/{n_windows} windows processed"
    )

    latest = df.iloc[-1]
    css_class = get_status_css(latest["status"])
    status_placeholder.markdown(
        f"<div class='status-card {css_class}'>Status: {latest['status']}</div>",
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = metrics_placeholder.columns(4)
    m1.metric("Latest MAE", f"{latest['mean_mae']:.6f}")
    m2.metric("Threshold", f"{latest['threshold']:.6f}")
    m3.metric("Violations", int(latest["violation_counter"]))
    m4.metric("Current Status", latest["status"])

    st.subheader("Live MAE vs Threshold")
    chart_df = df[["mean_mae", "threshold"]].copy()
    chart_placeholder.line_chart(chart_df)

    st.subheader("Recent Inference Results")
    table_placeholder.dataframe(
        df.tail(25),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total batches processed", len(df))
    s2.metric("Max MAE", f"{df['mean_mae'].max():.6f}")
    s3.metric("Alarm batches", int(df["alarm"].sum()))
    s4.metric("Sub-health batches", int(df["sub_health"].sum()))
else:
    st.info("No batches processed yet. Click 'Next Batch' or 'Run Full Cycle' to start.")