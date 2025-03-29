import pandas as pd 
import ollama
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from electra_battery_usage_market_prompt import *
from csv_analyzer import *

st.set_page_config(
    page_title="Battery LLM Pricing Indicator Dev",
    page_icon="⚡️",
    layout="wide"
)

st.title('💸 Battery Pricing Estimation Dev')

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file) if uploaded_file is not None else None

@st.cache_data
def get_cached_vehicle_usage_df(df):
    return get_vehicle_usage_df(df, generate_agg_fields_prompt)

# Initialize session state
if 'selected_vehicle' not in st.session_state:
    st.session_state.selected_vehicle = None    
if 'parameters' not in st.session_state:
    st.session_state.parameters = {}
if 'vehicle_params' not in st.session_state:
    st.session_state.vehicle_params = {}
if 'price_analysis_report' not in st.session_state:
    st.session_state.price_analysis_report = None

#Initialise session state for viz charts 
if 'pricing_comparison_fig' not in st.session_state:
    st.session_state.pricing_comparison_fig = None
if 'forecasting_behavior_fig' not in st.session_state:
    st.session_state.forecasting_behavior_fig = None
if 'battery_health_fig' not in st.session_state:
    st.session_state.battery_health_fig = None

col1, col2 = st.columns((1.5, 2), gap='large')

with col1:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility='collapsed')
    
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        vehicles_list = list(df['Topic' if 'Topic' in df.columns else 'vehicle_number'].unique())
        st.write(f"No. of vehicles in the source data: {len(vehicles_list)}")

        st.markdown("*Estimated time to run ~ 30-40 secs*")
        vehicle_usage_df = get_cached_vehicle_usage_df(df)    
    
        # Show dropdown only if vehicles exist
        if vehicles_list:
            selected_vehicle = st.selectbox(
                "Vehicle List", 
                vehicles_list,
                key='vehicle_selector',
                index=vehicles_list.index(st.session_state.selected_vehicle) if st.session_state.selected_vehicle in vehicles_list else 0
            )
            st.session_state.selected_vehicle = selected_vehicle  # Store in session state
                        
            if selected_vehicle:
                st.session_state.selected_vehicle = selected_vehicle
                vehicle_params = vehicle_usage_df[vehicle_usage_df['vehicle_number'] == selected_vehicle].iloc[0].to_dict()
                vehicle_params.pop('vehicle_number', None) 
                vehicle_params.pop('vehicle_summary')
                st.session_state.vehicle_params = vehicle_params  # Store in session state
                st.session_state.parameters = vehicle_params.copy()  # Default values for sidebar
                
        else:
            st.session_state.selected_vehicle = None  # Ensure it is reset
        
        # Sidebar Battery Usage Simulator
        st.sidebar.title("Battery Usage Simulator")
        
        usage_data = {}
        for key, value in st.session_state.parameters.items():
            usage_data[key] = st.sidebar.number_input(
                key.replace('_', ' ').title(), 
                value=value,
                step=0.1 if isinstance(value, float) else 1,
                key=key
            )
        
        st.session_state.vehicle_params.update(usage_data)  # Sync changes back
        
        if st.sidebar.button("Reset All"):
            st.session_state.parameters = st.session_state.vehicle_params.copy()
            st.rerun()
        
        if st.sidebar.button("Clear All"):
            st.session_state.selected_vehicle = None
            st.session_state.vehicle_params = {}
            st.session_state.parameters = {}
            st.session_state.price_analysis_report = None
            st.rerun()
    
        # st.subheader("Vehicle Usage Parameters", divider="orange")
        # st.json(st.session_state.vehicle_params)

        #     # Prevent error by checking if a vehicle is selected
        if st.session_state.selected_vehicle and st.button("Get Detailed Price Info for Selected Vehicle", icon="💰"):
            st.markdown("*GenAI is running & Calculating the Estimate..*")
            st.markdown("*Estimated time to run ~ 30-40 secs*")
            
            price_analysis_report = get_price_analysis_report(usage_data)
            st.session_state.price_analysis_report = price_analysis_report  # Store in session state
            
            #display the detailed report and forecasting chart for selected vehicle 
            if st.session_state.price_analysis_report:
                # st.subheader(f"💰 Price Forecasting", divider="green")
    
                # vehicle_id = st.session_state.selected_vehicle
                # price_final_dict = get_price_values(st.session_state.price_analysis_report)
                # fig = plot_price_forecasting_values(price_final_dict, vehicle_id)
                # st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("👩🏻‍💻 Price Analysis Full Report", divider="blue")
                price_summary_response = ollama.generate(
                    model='mistral', 
                    prompt=f"Present this report in a better tabular form: {st.session_state.price_analysis_report}"
                )
                st.write(price_summary_response['response'])

    if uploaded_file and not vehicle_usage_df.empty:
        if st.button("Battery Health Behavior Across Vehicles", icon="🔋", use_container_width=True):
            st.session_state.battery_health_fig = plot_battery_health_across_vehicles(vehicle_usage_df)
        
            if st.session_state.battery_health_fig:
                st.plotly_chart(st.session_state.battery_health_fig, use_container_width=True)

    if st.button("Get Battery Pricing Market Trends & Latest Updates", icon="💹", use_container_width=True):
        st.write("Gathering Battery Price News & Updates....", divider="green")
        latest_market_news_report = latest_market_news_headlines()
        st.write(latest_market_news_report)


def process_vehicle_forecast(i):
    """Function to process each vehicle separately."""
    usage_data = vehicle_usage_df['vehicle_summary'][i]
    vehicle_id = vehicle_usage_df['vehicle_number'][i]

    price_analysis_report = get_price_analysis_report(usage_data)
    price_final_dict = get_price_values(price_analysis_report)
    fig = plot_price_forecasting_values(price_final_dict, vehicle_id)
    return fig, vehicle_id
        
with col2:    
    st.subheader("Battery Price Comparison Across Vehicles", divider="blue")
    
    if uploaded_file and not vehicle_usage_df.empty:
        # if st.button("Show Pricing Comparison Across Vehicles", icon="🚙", use_container_width=True):
        st.markdown("*Estimated time to run ~ 2-3 mins*")
        all_vehicles_prices_fig = plot_prices_all_vehicles(vehicle_usage_df)
        st.session_state.pricing_comparison_fig = plot_prices_all_vehicles(vehicle_usage_df)

        if st.session_state.pricing_comparison_fig:
            st.plotly_chart(st.session_state.pricing_comparison_fig, use_container_width=True)

        if st.button("Forecasting Behavior Across Vehicles", icon="📉", use_container_width=True):
            st.write("Processing Battery Price Forecasting Chart...")
        
            num_vehicles = min(5, len(vehicle_usage_df))
        
            @st.cache_data
            def get_combined_forecasting_chart():
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(process_vehicle_forecast, range(num_vehicles)))
                
                figures, vehicle_ids = zip(*results)  # Unpack figures and vehicle IDs
                
                viridis_colors = pc.sequential.Plasma_r  # Get Viridis colors
                color_map = {i: viridis_colors[i % len(viridis_colors)] for i in range(num_vehicles)}
        
                combined_fig = go.Figure()
        
                for i, (fig, vehicle_id) in enumerate(zip(figures, vehicle_ids)):
                    for trace in fig['data']:
                        trace.name = f"Vehicle {vehicle_id}"  # Use actual vehicle ID
                        trace.line.color = color_map[i]  # Assign Viridis color
                        combined_fig.add_trace(trace)
        
                combined_fig.update_layout(
                    # template="plotly_dark",  # Optional: Use dark theme
                    title="Battery Price Forecasting Across Vehicles",
                    xaxis_title="Time Period",
                    yaxis_title="Forecasted Value (INR)",
                    legend_title="Vehicles",
                    height=600,
                    paper_bgcolor="black",  # Set entire background to white
                    plot_bgcolor="white",  # Set the plot area background to white
                    font=dict(color="black"),  # Ensure text is visible
                    yaxis=dict(
                        showgrid=False, 
                        # gridcolor="lightgrey",  # Light grey gridlines
                        zeroline=True
                    )
                )
        
                return combined_fig
        
            st.session_state.forecasting_behavior_fig = get_combined_forecasting_chart()
        
            if st.session_state.forecasting_behavior_fig:
                st.plotly_chart(st.session_state.forecasting_behavior_fig, use_container_width=True)

        