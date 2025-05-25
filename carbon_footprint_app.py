import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from test import CarbonEmissionPredictor
import plotly.graph_objects as go
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="🌱 Carbon Footprint Calculator",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #4CAF50, #2E7D32);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
    }
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'predicted_emission' not in st.session_state:
    st.session_state.predicted_emission = None

@st.cache_resource
def load_model():
    """Load the trained carbon emission prediction model"""
    try:
        predictor = CarbonEmissionPredictor('Carbon Emission.csv', force_retrain=False)
        if predictor.load_existing_model():
            # Load data for preprocessing consistency
            predictor.data = pd.read_csv('Carbon Emission.csv')
            predictor.preprocess_data()
            return predictor
        else:
            st.error("❌ Could not load the trained model. Please ensure 'carbon_emission_xgboost_model.pkl' exists.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def create_emission_gauge(emission_value):
    """Create a gauge chart for carbon emission"""
    # Define emission categories
    if emission_value < 1500:
        color = "green"
        category = "Low"
    elif emission_value < 3000:
        color = "yellow" 
        category = "Moderate"
    elif emission_value < 4500:
        color = "orange"
        category = "High"
    else:
        color = "red"
        category = "Very High"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = emission_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Carbon Footprint<br><span style='font-size:0.8em;color:gray'>{category} Emission Level</span>"},
        delta = {'reference': 2500},
        gauge = {
            'axis': {'range': [None, 6000]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 1500], 'color': "lightgreen"},
                {'range': [1500, 3000], 'color': "lightyellow"},
                {'range': [3000, 4500], 'color': "lightcoral"},
                {'range': [4500, 6000], 'color': "lightcoral"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 4500}}
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌱 Carbon Footprint Calculator</h1>
        <p>Calculate your personal carbon emissions based on lifestyle choices</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    predictor = load_model()
    if predictor is None:
        st.stop()
    
    # Sidebar for inputs
    st.sidebar.markdown("### 📋 Personal Information")
    
    # Personal characteristics
    with st.sidebar.expander("👤 Personal Details", expanded=True):
        body_type = st.selectbox("Body Type", ["normal", "overweight", "obese", "underweight"])
        sex = st.selectbox("Sex", ["male", "female"])
        diet = st.selectbox("Diet", ["omnivore", "vegetarian", "vegan", "pescatarian"])
        shower_freq = st.selectbox("How Often Do You Shower?", 
                                  ["daily", "twice a day", "more frequently", "less frequently"])
    
    # Home & Energy
    with st.sidebar.expander("🏠 Home & Energy", expanded=True):
        heating_energy = st.selectbox("Heating Energy Source", 
                                    ["natural gas", "electricity", "wood", "coal"])
        energy_efficiency = st.selectbox("Energy Efficiency Awareness", 
                                       ["Yes", "No", "Sometimes"])
    
    # Transportation
    with st.sidebar.expander("🚗 Transportation", expanded=True):
        transport = st.selectbox("Primary Transport", 
                                ["private", "public", "walk/bicycle"])
        vehicle_type = st.selectbox("Vehicle Type", 
                                  ["petrol", "diesel", "hybrid", "electric", "lpg", ""])
        vehicle_distance = st.number_input("Vehicle Monthly Distance (Km)", 
                                         min_value=0, max_value=15000, value=1000, step=50)
        air_travel = st.selectbox("Frequency of Air Travel", 
                                ["never", "rarely", "frequently", "very frequently"])
    
    # Lifestyle & Consumption
    with st.sidebar.expander("🛒 Lifestyle & Consumption", expanded=True):
        grocery_bill = st.number_input("Monthly Grocery Bill ($)", min_value = 0,value=200, step=10)
        social_activity = st.selectbox("Social Activity Level", 
                                     ["never", "sometimes", "often"])
        tv_hours = st.number_input("Daily TV/PC Hours", 
                                 min_value=0, max_value=24, value=4, step=1)
        internet_hours = st.number_input("Daily Internet Hours", 
                                       min_value=0, max_value=24, value=6, step=1)
        clothes_monthly = st.number_input("New Clothes per Month", 
                                        min_value=0, max_value=50, value=2, step=1)
    
    # Waste Management
    with st.sidebar.expander("🗑️ Waste Management", expanded=True):
        waste_bag_size = st.selectbox("Waste Bag Size", 
                                    ["small", "medium", "large", "extra large"])
        waste_bag_count = st.number_input("Waste Bags per Week", 
                                        min_value=1, max_value=10, value=3, step=1)
        
        # Recycling options
        st.markdown("**Recycling Materials:**")
        recycling_options = []
        if st.checkbox("Paper"):
            recycling_options.append("Paper")
        if st.checkbox("Plastic"):
            recycling_options.append("Plastic")
        if st.checkbox("Glass"):
            recycling_options.append("Glass")
        if st.checkbox("Metal"):
            recycling_options.append("Metal")
        
        recycling_str = str(recycling_options) if recycling_options else "[]"
    
    # Cooking Methods
    with st.sidebar.expander("🍳 Cooking Methods", expanded=True):
        st.markdown("**Cooking Equipment Used:**")
        cooking_options = []
        if st.checkbox("Stove"):
            cooking_options.append("Stove")
        if st.checkbox("Oven"):
            cooking_options.append("Oven")
        if st.checkbox("Microwave"):
            cooking_options.append("Microwave")
        if st.checkbox("Grill"):
            cooking_options.append("Grill")
        if st.checkbox("Air Fryer"):
            cooking_options.append("Airfryer")
        
        cooking_str = str(cooking_options) if cooking_options else "[]"
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prediction button
        if st.button("🔮 Calculate Carbon Footprint", type="primary", use_container_width=True):
            # Prepare input data
            sample_data = {
                'Body Type': body_type,
                'Sex': sex,
                'Diet': diet,
                'How Often Shower': shower_freq,
                'Heating Energy Source': heating_energy,
                'Transport': transport,
                'Vehicle Type': vehicle_type,
                'Social Activity': social_activity,
                'Monthly Grocery Bill': grocery_bill,
                'Frequency of Traveling by Air': air_travel,
                'Vehicle Monthly Distance Km': vehicle_distance,
                'Waste Bag Size': waste_bag_size,
                'Waste Bag Weekly Count': waste_bag_count,
                'How Long TV PC Daily Hour': tv_hours,
                'How Many New Clothes Monthly': clothes_monthly,
                'How Long Internet Daily Hour': internet_hours,
                'Energy efficiency': energy_efficiency,
                'Recycling': recycling_str,
                'Cooking_With': cooking_str
            }
            
            # Make prediction
            with st.spinner("🔄 Calculating your carbon footprint..."):
                predicted_emission = predictor.predict_new_sample(sample_data)
                
                if predicted_emission is not None:
                    st.session_state.predicted_emission = predicted_emission
                    st.session_state.prediction_made = True
                    st.success(f"✅ Prediction completed!")
                else:
                    st.error("❌ Failed to make prediction. Please check your inputs.")
    
    with col2:
        if st.session_state.prediction_made and st.session_state.predicted_emission is not None:
            emission = st.session_state.predicted_emission
            
            # Display gauge chart
            fig = create_emission_gauge(emission)
            st.plotly_chart(fig, use_container_width=True)
            
            # Emission value
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #2E7D32; margin: 0;">Annual Carbon Emission</h3>
                <h2 style="color: #1976D2; margin: 10px 0;">{emission:.0f} kg CO₂</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Results and recommendations
    if st.session_state.prediction_made and st.session_state.predicted_emission is not None:
        emission = st.session_state.predicted_emission
        
        st.markdown("---")
        st.markdown("### 📊 Your Carbon Footprint Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Comparison with averages
            avg_global = 4800  # Global average per person
            avg_us = 16000     # US average per person
            
            st.markdown("#### 🌍 Global Comparison")
            comparison_fig = go.Figure()
            comparison_fig.add_trace(go.Bar(
                x=['Your Footprint', 'Global Average', 'US Average'],
                y=[emission, avg_global, avg_us],
                marker_color=['#4CAF50', '#FFC107', '#F44336']
            ))
            comparison_fig.update_layout(
                title="Carbon Footprint Comparison",
                yaxis_title="CO₂ Emissions (kg/year)",
                height=300
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        with col2:
            # Environmental impact
            st.markdown("#### 🌳 Environmental Impact")
            trees_needed = emission / 22  # A tree absorbs ~22kg CO2/year
            st.metric("Trees needed to offset", f"{trees_needed:.0f} trees")
            
            # Equivalent comparisons
            car_miles = emission / 0.4  # ~0.4 kg CO2 per mile
            st.metric("Equivalent car miles", f"{car_miles:.0f} miles")
            
            flights = emission / 500  # ~500 kg CO2 per domestic flight
            st.metric("Equivalent domestic flights", f"{flights:.1f} flights")
        
        with col3:
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            
            if emission > 3000:
                recommendations = [
                    "🚗 Consider using public transport more",
                    "⚡ Switch to renewable energy",
                    "🥗 Reduce meat consumption",
                    "♻️ Increase recycling efforts",
                    "🏠 Improve home energy efficiency"
                ]
            elif emission > 2000:
                recommendations = [
                    "🚲 Use bicycle for short trips",
                    "💡 Use energy-efficient appliances",
                    "♻️ Expand recycling to all materials",
                    "🌱 Consider plant-based meals",
                    "📱 Reduce screen time"
                ]
            else:
                recommendations = [
                    "🌟 Great job! Keep it up!",
                    "📢 Share your practices with others",
                    "🌱 Consider carbon offset programs",
                    "📚 Continue learning about sustainability",
                    "👥 Advocate for environmental policies"
                ]
            
            for rec in recommendations:
                st.markdown(f"• {rec}")
        
        # Detailed breakdown
        st.markdown("---")
        st.markdown("### 📈 Emission Category Breakdown")
        
        # Create a sample breakdown (this would ideally come from the model)
        categories = ['Transportation', 'Home Energy', 'Food & Diet', 'Consumption', 'Waste']
        
        # Estimate breakdown based on inputs (simplified logic)
        transport_emission = vehicle_distance * 0.2 + (air_travel == "very frequently") * 500
        home_emission = (heating_energy in ["coal", "natural gas"]) * 400 + tv_hours * 10
        food_emission = {"omnivore": 600, "vegetarian": 400, "pescatarian": 450, "vegan": 300}[diet]
        consumption_emission = grocery_bill * 2 + clothes_monthly * 50
        waste_emission = waste_bag_count * 20
        
        # Normalize to actual prediction
        total_estimated = transport_emission + home_emission + food_emission + consumption_emission + waste_emission
        scale_factor = emission / total_estimated if total_estimated > 0 else 1
        
        breakdown_values = [
            transport_emission * scale_factor,
            home_emission * scale_factor,
            food_emission * scale_factor,
            consumption_emission * scale_factor,
            waste_emission * scale_factor
        ]
        
        # Create pie chart
        pie_fig = px.pie(
            values=breakdown_values,
            names=categories,
            title="Carbon Emission Breakdown by Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        pie_fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(pie_fig, use_container_width=True)
        
        # Summary table
        st.markdown("#### 📋 Emission Summary by Category")
        breakdown_df = pd.DataFrame({
            'Category': categories,
            'Emissions (kg CO₂)': [f"{val:.0f}" for val in breakdown_values],
            'Percentage': [f"{val/emission*100:.1f}%" for val in breakdown_values]
        })
        st.dataframe(breakdown_df, use_container_width=True)

if __name__ == "__main__":
    main()
