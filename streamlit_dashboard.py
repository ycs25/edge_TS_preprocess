import os
import time
import sqlite3
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Edge-Cloud CNC Anomaly Dashboard",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
.main-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
.sub-text { color: #b0b0b0; margin-bottom: 1rem; }
.status-card {
    padding: 16px; border-radius: 12px; text-align: center; font-size: 22px;
    font-weight: 700; color: white; margin-top: 10px; margin-bottom: 10px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.healthy { background: linear-gradient(90deg, #1faa59, #2ecc71); }
.warning { background: linear-gradient(90deg, #d4ac0d, #f1c40f); color: black; }
.alarm { background: linear-gradient(90deg, #c0392b, #e74c3c); animation: blink 1s infinite; }
@keyframes blink { 50% { opacity: 0.45; } }
</style>
""", unsafe_allow_html=True)

# PATHS of database and data
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT_DIR, "data", "processed", "inference_history.db")

def get_sigma_bounds(source_file):
    """
    Load 3-sigma & 4-sigma from selected_file for slider boundries.
    If not found, return default values.
    """
    default_low, default_high = 0.3, 0.6
    if not os.path.exists(DB_PATH):
        return default_low, default_high
    
    conn = sqlite3.connect(DB_PATH)
    try:
        if source_file == "All":
            query = "SELECT threshold_3sigma, threshold_4sigma FROM history LIMIT 1"
            res = pd.read_sql_query(query, conn)
        else:
            query = "SELECT threshold_3sigma, threshold_4sigma FROM history WHERE source_file = ? LIMIT 1"
            res = pd.read_sql_query(query, conn, params=[source_file])
        
        if not res.empty:
            return float(res.iloc[0]['threshold_3sigma']), float(res.iloc[0]['threshold_4sigma'])
    except Exception as e:
        print(f"Error fetching sigma bounds: {e}")
    finally:
        conn.close()
    return default_low, default_high

@st.cache_data(ttl=5) # cache for 5 seconds
def fetch_full_history(source_file):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM history"
    params = []
    
    if source_file and source_file != "All":
        query += " WHERE source_file = ?"
        params.append(source_file)
        
    # Retrieve data ordered by timestamp to ensure correct playback sequence
    query += " ORDER BY timestamp ASC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def fetch_available_sources():
    if not os.path.exists(DB_PATH):
        return ["All"]
    conn = sqlite3.connect(DB_PATH)
    try:
        sources = pd.read_sql_query("SELECT DISTINCT source_file FROM history", conn)["source_file"].tolist()
        return ["All"] + sources
    except Exception:
        return ["All"]
    finally:
        conn.close()

# --- 1. Session State ---
# Initialize playback index, current source, and play/pause state in session state
if "playback_index" not in st.session_state:
    st.session_state.playback_index = 0
if "current_source" not in st.session_state:
    st.session_state.current_source = None
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

st.markdown("<div class='main-title'>CNC Anomaly Detection Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Playback & Live Monitoring Mode</div>", unsafe_allow_html=True)

# --- 2. Sidebar: Controls ---
with st.sidebar:
    st.header("🗂️ Data Source")
    available_sources = fetch_available_sources()
    selected_source = st.selectbox("Select Source File to Play", options=available_sources)
    
    # Detect when the user switches the data source, automatically reset the playback pointer and start playing
    if selected_source != st.session_state.current_source:
        st.session_state.current_source = selected_source
        st.session_state.playback_index = 1
        st.session_state.is_playing = True
    
    t3, t4 = get_sigma_bounds(selected_source)
    st.markdown("### ⚠️ Alarm Logic Parameters")

    slider_min = min(t3, t4)
    slider_max = max(t3, t4)
    if slider_min == slider_max:
        slider_max = slider_min + 0.1

    custom_threshold = st.slider(
        "Absolute MAE Threshold", 
        min_value=float(slider_min), 
        max_value=float(slider_max), 
        value=float((slider_min + slider_max) / 2), 
        step=0.001,
        format="%.4f"
    )

    debounce_n = st.number_input("Debounce limit", min_value=1, max_value=10, value=3)

    st.divider()
    st.header("▶️ Playback Controls")
    
    col1, col2, col3 = st.columns(3)
    if col1.button("▶️ Play"):
        st.session_state.is_playing = True
    if col2.button("⏸️ Pause"):
        st.session_state.is_playing = False
    if col3.button("🔄 Reset"):
        st.session_state.playback_index = 1
        st.session_state.is_playing = True
        
    playback_speed = st.slider("Playback Speed (Windows/tick)", 1, 10, 2)
    refresh_rate = st.slider("Refresh Interval (sec)", 0.1, 1.0, 0.2, step=0.1)
    display_limit = st.number_input("Display Window Size (Scrolling)", min_value=50, max_value=200, value=100, step=50)



# --- 3. Fetch Full History ---
df_full = fetch_full_history(selected_source)

if df_full.empty:
    st.info("⏳ Waiting for data... Start `app.py` and `edge_simulator.py` to ingest data.")
else:
    total_rows = len(df_full)
    
    # Make sure playback index does not exceed total rows, if it does, reset to total and pause
    if st.session_state.playback_index > total_rows:
        st.session_state.playback_index = total_rows
        st.session_state.is_playing = False

    # Slice the dataframe up to the current playback index to simulate streaming data
    df_current_history = df_full.iloc[:st.session_state.playback_index]
    
    # UI on top of the page to show playback progress
    st.progress(st.session_state.playback_index / total_rows, text=f"Playback Progress: {st.session_state.playback_index} / {total_rows} Windows")

    # --- 4. Business Logic Calculation (Based on Current Progress) ---
    if not df_current_history.empty:
        # Calculate violation counter
        recent_windows = df_current_history["window_error"].tolist()
        violation_counter = 0
        for error in reversed(recent_windows):
            if error > custom_threshold:
                violation_counter += 1
            else:
                break
                
        is_alarm = violation_counter >= debounce_n
        is_warning = violation_counter > 0 and not is_alarm

        if is_alarm:
            status_text, css_class = "🔴 Critical Anomaly", "alarm"
        elif is_warning:
            status_text, css_class = "🟡 Warning (Debouncing)", "warning"
        else:
            status_text, css_class = "🟢 Healthy", "healthy"

        # Render status card
        st.markdown(f"<div class='status-card {css_class}'>{status_text}</div>", unsafe_allow_html=True)

        # Render metrics
        latest = df_current_history.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest Window Error", f"{latest['window_error']:.4f}")
        m2.metric("Custom Threshold", f"{custom_threshold:.2f}")
        m3.metric("Consecutive Violations", f"{violation_counter} / {debounce_n}")
        m4.metric("Current File", selected_source)

        # --- 5. Render Chart (Implement Scrolling Window Effect) ---
        st.subheader("📈 Streaming Window Error Chart")
        
        # Only take the latest display_limit records for plotting, creating a left-scrolling effect    
        df_display = df_current_history.tail(display_limit).copy()
        df_display["Threshold"] = custom_threshold
        df_display["Time"] = pd.to_datetime(df_display["timestamp"], unit="s").dt.strftime('%H:%M:%S.%f').str[:-3]
        df_display = df_display.set_index("Time")
        st.line_chart(df_display[["window_error", "Threshold"]], color=["#3498db", "#e74c3c"])

        # --- 6. Auto-playback Logic ---
        if st.session_state.is_playing and st.session_state.playback_index < total_rows:
            # Increment playback index by the selected speed, but do not exceed total rows
            st.session_state.playback_index += playback_speed
            time.sleep(refresh_rate)
            st.rerun()