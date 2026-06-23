import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import datetime

# Import our backend modules
from model import prepare_features
from optimizer import optimize_schedule

# ----------------------------------------------------
# 1. Page Configuration & Custom CSS Injection
# ----------------------------------------------------
st.set_page_config(
    page_title="VoltShift: Shanvi's Green Energy Saver",
    layout="wide"
)

# Custom High-End Cyberpunk Theme styling (Slate, Glassmorphism, Neon Cyan & Green Glow)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Global Dark Backdrop Override */
    html, body, [class*="css"], .stApp {
        background-color: #070708 !important;
        color: #FAFAFA !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Minimize bulk padding top/bottom */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Sidebar Styling (Deep charcoal black) */
    [data-testid="stSidebar"] {
        background-color: #040405 !important;
        border-right: 1px solid #141416 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {
        color: #94A3B8 !important;
        font-weight: 500;
    }

    /* Status Bar Badge style */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(0, 255, 135, 0.08);
        border: 1px solid rgba(0, 255, 135, 0.2);
        color: #00FF87;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 4px 10px;
        border-radius: 20px;
        margin-bottom: 12px;
    }

    .tech-banner {
        font-size: 11px;
        color: #475569;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 25px;
        border-bottom: 1px solid #141416;
        padding-bottom: 12px;
    }

    /* Custom Glassmorphism Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(20, 20, 22, 0.8) 0%, rgba(12, 12, 14, 0.8) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
    }

    /* Glowing ambient card border on hover */
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0, 245, 255, 0.12);
        border-color: rgba(0, 245, 255, 0.25);
    }

    .metric-title {
        font-size: 11px;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 34px;
        color: #FAFAFA;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: -0.03em;
    }

    .metric-delta {
        font-size: 13px;
        font-weight: 600;
    }

    .metric-sub {
        font-size: 11px;
        color: #475569;
        margin-top: 4px;
    }

    /* Equivalency badges style */
    .equivalent-box {
        background-color: #0F0F11;
        border: 1px solid #18181C;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 12px;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 28px;
        border-bottom: 1px solid #141416;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent !important;
        color: #64748B !important;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        color: #00F5FF !important;
        border-bottom: 2px solid #00F5FF !important;
    }

    /* Custom Sliders styling */
    .stSlider > div [data-baseweb="slider"] {
        background-color: #141416 !important;
    }
    .stSlider > div [data-baseweb="slider-thumb"] {
        background-color: #00F5FF !important;
        box-shadow: 0 0 10px #00F5FF !important;
        border: 2px solid #00F5FF !important;
    }

    /* Custom panels */
    .insight-box {
        background: rgba(15, 15, 17, 0.5);
        border: 1px solid #141416;
        border-radius: 12px;
        padding: 20px;
        height: 100%;
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 1.1 Glowing Status Badge & Headers
st.markdown("<div class='status-badge'>● Operational // Node Active</div>", unsafe_allow_html=True)
st.markdown("<h1 style='color: #FAFAFA; margin-bottom: 0px; font-weight: 700; letter-spacing: -0.02em; font-size: 34px;'>VOLTSHIFT // Smart Energy Saver</h1>", unsafe_allow_html=True)
st.markdown("<div class='tech-banner'>Model Accuracy: 99.55% &nbsp;//&nbsp; Solver Convergence: 50ms &nbsp;//&nbsp; Facility Node: Shanvi</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Sidebar Parameters & Developer Profile
# ----------------------------------------------------
st.sidebar.markdown("<h3 style='color: #FAFAFA; margin-top: 0px; letter-spacing: 0.05em;'>Controls</h3>", unsafe_allow_html=True)

weight_cost = st.sidebar.slider(
    "What is your priority?",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Slide left to prioritize carbon reduction (Earth). Slide right to prioritize saving money (Wallet)."
)

# Custom Labeling based on Slider position (Earth vs Wallet)
bias_label = "Balanced (Earth & Wallet)"
if weight_cost >= 0.8:
    bias_label = "Priority: Save Money"
elif weight_cost <= 0.2:
    bias_label = "Priority: Save the Planet"
st.sidebar.caption(f"Priority State: **{bias_label}**")

st.sidebar.markdown("<div style='height: 1px; background: #141416; margin: 20px 0;'></div>", unsafe_allow_html=True)

temp_offset = st.sidebar.slider(
    "Simulate Weather Changes (°C)",
    min_value=-5.0,
    max_value=5.0,
    value=0.0,
    step=0.5,
    help="Add or subtract degrees to simulate a sudden summer heatwave or winter cold snap."
)

selected_date = st.sidebar.date_input(
    "Pick a Day of the Year",
    value=datetime.date(2025, 6, 15),
    min_value=datetime.date(2025, 1, 1),
    max_value=datetime.date(2025, 12, 30)
)

# PERSONAL DEVELOPER SIDEBAR PANEL
st.sidebar.markdown("""
<div style="background-color: #0C0C0E; border: 1px solid #141416; border-radius: 12px; padding: 16px; margin-top: 30px;">
    <h4 style="color: #00F5FF; margin-top: 0px; margin-bottom: 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;">Project Developer</h4>
    <p style="color: #FAFAFA; font-size: 14px; font-weight: 700; margin-bottom: 2px;">Shanvi</p>
    <p style="color: #71717A; font-size: 12px; margin-bottom: 12px;">GitHub: <a href="https://github.com/shvi2103" target="_blank" style="color: #00FF87; text-decoration: none; font-weight: 600;">@shvi2103</a></p>
    <p style="color: #64748B; font-size: 11px; line-height: 1.4; margin-bottom: 0px;">
        Designed to show how time-series ML modeling and numerical optimization (SciPy) can balance grid stability and corporate utility expenses.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 3. Load Model and Data
# ----------------------------------------------------
@st.cache_resource
def load_ml_model():
    return joblib.load("energy_forecast_model.pkl")

@st.cache_data
def load_dataset():
    df = pd.read_csv("building_energy_data.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

try:
    model = load_ml_model()
    df_raw = load_dataset()
except FileNotFoundError:
    st.error("Missing required model or dataset files! Run both 'data_generator.py' and 'model.py' first.")
    st.stop()

# ----------------------------------------------------
# 4. Data Processing & ML Forecasting
# ----------------------------------------------------
target_date_str = selected_date.strftime("%Y-%m-%d")
day_df = df_raw[df_raw['Timestamp'].dt.strftime("%Y-%m-%d") == target_date_str].copy()

if len(day_df) != 24:
    st.warning("Please choose a date between 2025-01-01 and 2025-12-30.")
    st.stop()

day_df = prepare_features(day_df)
day_df['Temperature'] = day_df['Temperature'] + temp_offset

# Predict base load using Shanvi's trained model
feature_cols = ['Temperature', 'Hour', 'DayOfWeek', 'Month', 'IsWeekday']
day_df['Predicted_Load'] = model.predict(day_df[feature_cols])

baseline_loads = day_df['Predicted_Load'].values
prices = day_df['Electricity_Price'].values
carbon_intensities = day_df['Grid_Carbon_Intensity'].values

# Run SciPy SLSQP Solver
optimized_loads, optimized_flexible = optimize_schedule(
    baseline_loads, prices, carbon_intensities, weight_cost
)

day_df['Optimized_Load'] = optimized_loads

# ----------------------------------------------------
# 5. Core Metric Calculations
# ----------------------------------------------------
base_cost = np.sum(baseline_loads * prices)
base_carbon = np.sum(baseline_loads * carbon_intensities) / 1000.0 # kg

opt_cost = np.sum(optimized_loads * prices)
opt_carbon = np.sum(optimized_loads * carbon_intensities) / 1000.0 # kg

cost_savings = base_cost - opt_cost
carbon_savings = base_carbon - opt_carbon

cost_pct = (cost_savings / base_cost) * 100 if base_cost > 0 else 0
carbon_pct = (carbon_savings / base_carbon) * 100 if base_carbon > 0 else 0

peak_base = np.max(baseline_loads)
peak_opt = np.max(optimized_loads)
peak_reduction = ((peak_base - peak_opt) / peak_base) * 100

# Environmental Equivalents Calculations (Translating data into human-visuals)
tree_days = (carbon_savings / 0.06) # 1 tree absorbs ~0.06kg CO2 per day
miles_saved = (carbon_savings * 2.5) # ~2.5 miles of average car driving per kg CO2

# ----------------------------------------------------
# 6. Tab Definitions (Now featuring Developer Journal)
# ----------------------------------------------------
tab_control, tab_xai, tab_journal = st.tabs(["Control Room", "How the AI Decides", "Developer Journal"])

# ====================================================
# TAB 1: CONTROL ROOM (DASHBOARD)
# ====================================================
with tab_control:
    col1, col2, col3 = st.columns(3)

    # Simplified, friendly metric cards
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Money Saved Today</div>
            <div class="metric-value">${cost_savings:.2f}</div>
            <div class="metric-delta" style="color: #00FF87;">↓ {cost_pct:.1f}% Cost Reduction</div>
            <div class="metric-sub">Base cost: ${base_cost:.2f} | Smart cost: ${opt_cost:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Carbon Prevented</div>
            <div class="metric-value">{carbon_savings:.2f} kg CO₂</div>
            <div class="metric-delta" style="color: #00F5FF;">↓ {carbon_pct:.1f}% Clean Energy</div>
            <div class="metric-sub">Base footprint: {base_carbon:.1f} kg | Smart: {opt_carbon:.1f} kg</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peak Power Reduction</div>
            <div class="metric-value">{peak_reduction:.1f}%</div>
            <div class="metric-delta" style="color: #00FF87;">↓ {peak_base - peak_opt:.1f} kW load shed</div>
            <div class="metric-sub">Protects municipal grids from blackout overload</div>
        </div>
        """, unsafe_allow_html=True)

    # Environmental Equivalency Indicators
    col_e1, col_e2, col_spacer = st.columns([1, 1, 1])
    with col_e1:
        st.markdown(f"""
        <div class="equivalent-box">
            <span style="color: #00FF87; font-weight: 700; font-size: 16px;">🌲</span>
            <span>Offset equivalent to <b>{tree_days:.1f} tree-days</b> of carbon absorption.</span>
        </div>
        """, unsafe_allow_html=True)
    with col_e2:
        st.markdown(f"""
        <div class="equivalent-box">
            <span style="color: #00F5FF; font-weight: 700; font-size: 16px;">🚗</span>
            <span>Prevented <b>{miles_saved:.1f} miles</b> of standard gasoline car emissions.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Glowing Area Shaded Comparison Graph (Single axis, extremely professional)
    fig = go.Figure()

    # Muted, clean dashed grey for baseline (Normal use)
    fig.add_trace(go.Scatter(
        x=day_df['Hour'], 
        y=day_df['Predicted_Load'],
        mode='lines',
        name='Running AC Normally (Baseline)',
        line=dict(color='#475569', width=2.5, dash='dash'),
        hovertemplate="Normal: %{y:.1f} kW<extra></extra>"
    ))

    # Glowing, bold neon green with gorgeous soft translucent fill under the curve
    fig.add_trace(go.Scatter(
        x=day_df['Hour'], 
        y=day_df['Optimized_Load'],
        mode='lines+markers',
        name="Shanvi's Smart Energy Schedule (Optimized)",
        line=dict(color='#00FF87', width=3.5),
        marker=dict(size=6, color='#00FF87'),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 135, 0.03)', # Beautiful glowing gradient fill under line
        hovertemplate="Smart: %{y:.1f} kW<extra></extra>"
    ))

    fig.update_layout(
        title="Comparison: Standard Power vs. Smart Shifting Power",
        title_font=dict(color="#FAFAFA", size=15),
        xaxis=dict(
            title="Time of Day (Hour)",
            title_font=dict(color="#64748B", size=12),
            tickmode='linear',
            tick0=0,
            dtick=2,
            gridcolor='#141416',
            tickfont=dict(color="#64748B"),
            zeroline=False
        ),
        yaxis=dict(
            title="Power Consumed (Kilowatts)",
            title_font=dict(color="#64748B", size=12),
            gridcolor='#141416',
            tickfont=dict(color="#64748B"),
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            font=dict(color="#64748B", size=11)
        ),
        plot_bgcolor='#070708',
        paper_bgcolor='#070708',
        margin=dict(l=40, r=40, t=40, b=40),
        height=420,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Details & Table Layout
    col_ins, col_tbl = st.columns([1, 1])

    with col_ins:
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #FAFAFA; margin-top: 0px;'>What is the App doing right now?</h4>", unsafe_allow_html=True)
        
        cheapest_hours = np.argsort(prices)[:3]
        expensive_hours = np.argsort(prices)[-3:]
        lowest_carbon_hours = np.argsort(carbon_intensities)[:3]

        if weight_cost > 0.3:
            st.markdown(f"""
            <p style='color: #94A3B8; font-size: 13.5px; line-height: 1.6;'>
            <b>Save Money Plan:</b><br>
            The app noticed that electricity rates are highest in the afternoon at <b>{', '.join([f'{h}:00' for h in expensive_hours])}</b>. 
            It automatically turned down the AC during those hours and pre-cooled the building in the morning at <b>{', '.join([f'{h}:00' for h in cheapest_hours])}</b> when electricity is extremely cheap.
            </p>
            """, unsafe_allow_html=True)
        if weight_cost < 0.7:
            st.markdown(f"""
            <p style='color: #94A3B8; font-size: 13.5px; line-height: 1.6; margin-top: 15px;'>
            <b>Save the Planet Plan:</b><br>
            The app shifted power usage into the afternoon hours around <b>{', '.join([f'{h}:00' for h in lowest_carbon_hours])}</b>. 
            This is when the sun is blazing and solar energy on the electrical grid is at its absolute maximum, making your power run on clean energy.
            </p>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_tbl:
        st.markdown("<h4 style='color: #FAFAFA; margin-top: 0px;'>Hourly Details Table</h4>", unsafe_allow_html=True)
        
        # HTML Custom Table
        table_df = day_df[['Hour', 'Temperature', 'Electricity_Price', 'Grid_Carbon_Intensity', 'Predicted_Load', 'Optimized_Load']].copy()
        
        rows_html = ""
        for idx, row in table_df.iterrows():
            hr = int(row['Hour'])
            temp = row['Temperature']
            price = row['Electricity_Price']
            carb = row['Grid_Carbon_Intensity']
            pred = row['Predicted_Load']
            opt = row['Optimized_Load']
            
            rows_html += f"""
            <tr style="border-bottom: 1px solid #141416;">
                <td style="padding: 10px 12px; color: #475569;">{hr:02d}:00</td>
                <td style="padding: 10px 12px; color: #FAFAFA;">{temp:.1f}°C</td>
                <td style="padding: 10px 12px; color: #00F5FF;">${price:.2f}</td>
                <td style="padding: 10px 12px; color: #94A3B8;">{carb:.0f}g</td>
                <td style="padding: 10px 12px; color: #F87171; font-weight: 500;">{pred:.1f} kW</td>
                <td style="padding: 10px 12px; color: #00FF87; font-weight: 600;">{opt:.1f} kW</td>
            </tr>
            """
            
        custom_table = f"""
        <div style="max-height: 230px; overflow-y: auto; border: 1px solid #141416; border-radius: 12px; background-color: #0A0A0B;">
            <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 12.5px;">
                <thead>
                    <tr style="background-color: #040405; border-bottom: 1px solid #141416; position: sticky; top: 0; z-index: 10;">
                        <th style="padding: 12px; color: #475569; font-weight: 600;">Hour</th>
                        <th style="padding: 12px; color: #475569; font-weight: 600;">Temp</th>
                        <th style="padding: 12px; color: #475569; font-weight: 600;">Price</th>
                        <th style="padding: 12px; color: #475569; font-weight: 600;">Carbon</th>
                        <th style="padding: 12px; color: #475569; font-weight: 600;">Standard AC</th>
                        <th style="padding: 12px; color: #475569; font-weight: 600;">Smart AC</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
        st.markdown(custom_table, unsafe_allow_html=True)

# ====================================================
# TAB 2: HOW THE AI DECIDES
# ====================================================
with tab_xai:
    st.markdown("<h3 style='color: #FAFAFA; margin-top: 10px;'>How the AI Decides</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #94A3B8; font-size: 14px; line-height: 1.6;'>
    Before we can plan tomorrow's schedule, the system has to predict exactly how much electricity the building will need. 
    Our AI model does this by looking at weather forecasts and calendars. 
    Below, you can see <b>what the AI cares about most</b> when making its predictions:
    </p>
    """, unsafe_allow_html=True)
    
    # Fetch & Sort Feature Importances from trained model
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    # Friendly labels for the features
    friendly_feature_map = {
        'Temperature': 'Outdoor Temperature (HVAC Needs)',
        'Hour': 'Time of Day (Working Hours)',
        'DayOfWeek': 'Day of the Week (Weekday vs. Weekend)',
        'Month': 'Month of the Year (Seasonal Climate)',
        'IsWeekday': 'Is it a Workday? (Occupancy)'
    }
    
    sorted_names = [friendly_feature_map[feature_cols[i]] for i in indices]
    sorted_importances = importances[indices]
    
    # Render Glowing Horizontal Bar Chart (glowing electric cyan)
    fig_imp = go.Figure()
    
    fig_imp.add_trace(go.Bar(
        x=sorted_importances,
        y=sorted_names,
        orientation='h',
        marker=dict(
            color='rgba(0, 245, 255, 0.1)',
            line=dict(color='#00F5FF', width=2)
        ),
        name="Importance Weight"
    ))
    
    fig_imp.update_layout(
        xaxis=dict(
            title="Influence Weight (How much the AI relies on this feature)",
            title_font=dict(color="#64748B", size=12),
            gridcolor='#141416',
            tickfont=dict(color="#475569"),
            zeroline=False
        ),
        yaxis=dict(
            tickfont=dict(color="#94A3B8")
        ),
        plot_bgcolor='#070708',
        paper_bgcolor='#070708',
        margin=dict(l=100, r=40, t=20, b=40),
        height=320
    )
    
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("<h4 style='color: #FAFAFA;'>What does this chart mean?</h4>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <ul style="color: #94A3B8; font-size: 14px; line-height: 1.8;">
        <li><b>Weather is King:</b> The chart shows that <b>Outdoor Temperature</b> is the biggest factor. This is because heating and cooling are the massive power hogs of any building.</li>
        <li><b>Office Schedules:</b> The <b>Time of Day</b> and <b>Is it a Workday?</b> are also huge factors because the building is mostly empty and quiet during nights and weekends.</li>
    </ul>
    """, unsafe_allow_html=True)

# ====================================================
# TAB 3: SHANVI'S DEVELOPER JOURNAL
# ====================================================
with tab_journal:
    st.markdown("<h3 style='color: #FAFAFA; margin-top: 10px;'>Shanvi's Developer Journal</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #0C0C0E; border: 1px solid #141416; border-radius: 12px; padding: 20px; color: #94A3B8; font-size: 14px; line-height: 1.8;">
        <h4 style="color: #00F5FF; margin-top: 0px;">Why I built Project VoltShift</h4>
        <p>
            Large office buildings are massive energy consumers. They run their AC and heating in a reactive way—turning them up full blast on hot afternoons, which is exactly when electricity is most expensive and power plants burn the most coal.
            I built VoltShift to show that we can make a building "grid-smart" by forecasting tomorrow's demand and shifting energy loads to cleaner, cheaper hours automatically.
        </p>
        
        <h4 style="color: #00F5FF; margin-top: 20px;">My Engineering Choices & Decisions</h4>
        <ul>
            <li>
                <b>Why I chose Random Forest:</b> 
                I initially considered complex neural networks (LSTMs) for forecasting building loads. However, for a single building's tabular features, 
                a simple Random Forest Regressor trains in <b>under 1 second</b> on my CPU while securing an outstanding <b>99.55% accuracy score</b>. 
                Choosing this model keeps the code lightweight, easily deployable, and computationally efficient.
            </li>
            <li>
                <b>Why I chose SciPy SLSQP Solver:</b> 
                For the optimization part, we have a strict requirement: the building must get the exact amount of cooling/power it needs for the day (we can't just turn off the AC and let people sweat!). 
                The SLSQP algorithm successfully handles this <b>equality constraint</b>, shifting the loads gracefully and calculating the results in just 50 milliseconds.
            </li>
        </ul>
        
        <h4 style="color: #00F5FF; margin-top: 20px;">My Roadmap for Future Enhancements (TODOs)</h4>
        <ol>
            <li><b>Live Grid Connections:</b> Hook the optimizer up to real, live utility pricing feeds instead of simulated vectors.</li>
            <li><b>Physical Hardware Control:</b> Integrate BACnet protocols to send these optimized target schedules directly to active building AC controllers.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)