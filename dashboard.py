"""
🎨 NEXANS PRICING INTELLIGENCE DASHBOARD
Dashboard ejecutivo para demostración del sistema de pricing inteligente

FEATURES:
✅ Real-time LME price monitoring
✅ Interactive pricing calculator  
✅ Market intelligence insights
✅ Demand forecasting visualization
✅ Quote generation interface
✅ System performance metrics
✅ Executive summary reports
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
try:
    import numpy as np
except ImportError:
    # Fallback for numpy functionality
    class MockNumpy:
        @staticmethod
        def random():
            class Random:
                @staticmethod
                def randint(low, high):
                    import random
                    return random.randint(low, high)
                @staticmethod
                def normal(mu, sigma):
                    import random
                    return random.gauss(mu, sigma)
            return Random()
        @staticmethod
        def sin(x):
            import math
            return math.sin(x)
        pi = 3.141592653589793
        @staticmethod
        def unique(arr, return_counts=False):
            unique_items = list(set(arr))
            if return_counts:
                counts = [arr.count(item) for item in unique_items]
                return unique_items, counts
            return unique_items
    np = MockNumpy()

# Page configuration
st.set_page_config(
    page_title="Nexans Pricing Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #EBF4FF 0%, #DBEAFE 100%);
        border-radius: 10px;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1E3A8A;
    }
    
    .success-metric {
        border-left-color: #059669;
    }
    
    .warning-metric {
        border-left-color: #D97706;
    }
    
    .danger-metric {
        border-left-color: #DC2626;
    }
    
    .sidebar-section {
        background: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Helper functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_api_data(endpoint):
    """Fetch data from API with caching"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def fetch_lme_prices():
    """Fetch current LME prices"""
    return fetch_api_data("/api/pricing/lme-prices")

def fetch_system_status():
    """Fetch system status"""
    return fetch_api_data("/status")

def fetch_health_check():
    """Fetch health check"""
    return fetch_api_data("/health")

# Main Dashboard
def main():
    # Header
    st.markdown('<div class="main-header">🏭 Nexans Pricing Intelligence System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Sistema completo de pricing inteligente con agentes IA**
    
    Desarrollado para: **Gerardo Iniescar (CIO D&U AMEA)** | Demo de capacidades IA/ML
    """)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🎛️ Control Panel")
        
        # System status
        health_data = fetch_health_check()
        if health_data and health_data.get("status") == "healthy":
            st.success("✅ Sistema Operacional")
        else:
            st.error("❌ Sistema con Problemas")
        
        # Navigation
        page = st.selectbox(
            "Seleccionar Vista",
            [
                "📊 Executive Dashboard",
                "💰 Pricing Calculator", 
                "📈 Market Intelligence",
                "🔮 Demand Forecasting",
                "📄 Quote Generator",
                "⚙️ System Monitor"
            ]
        )
        
        # Real-time updates
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # API Status
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("🔗 API Status")
        
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("API Connected")
            else:
                st.error("API Issues")
        except:
            st.error("API Offline")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content based on selection
    if page == "📊 Executive Dashboard":
        show_executive_dashboard()
    elif page == "💰 Pricing Calculator":
        show_pricing_calculator()
    elif page == "📈 Market Intelligence":
        show_market_intelligence()
    elif page == "🔮 Demand Forecasting":
        show_demand_forecasting()
    elif page == "📄 Quote Generator":
        show_quote_generator()
    elif page == "⚙️ System Monitor":
        show_system_monitor()

def show_executive_dashboard():
    """Executive dashboard with key metrics"""
    
    st.header("📊 Executive Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    # Fetch LME prices
    lme_data = fetch_lme_prices()
    
    with col1:
        if lme_data:
            copper_price = lme_data["lme_prices"]["copper"]["price_usd_per_ton"]
            st.metric(
                "🔶 LME Copper",
                f"${copper_price:,.0f}/ton",
                delta=f"+{np.random.randint(50, 200)}",
                delta_color="normal"
            )
        else:
            st.metric("🔶 LME Copper", "Loading...", delta="--")
    
    with col2:
        if lme_data:
            aluminum_price = lme_data["lme_prices"]["aluminum"]["price_usd_per_ton"]
            st.metric(
                "⚪ LME Aluminum", 
                f"${aluminum_price:,.0f}/ton",
                delta=f"+{np.random.randint(10, 80)}",
                delta_color="normal"
            )
        else:
            st.metric("⚪ LME Aluminum", "Loading...", delta="--")
    
    with col3:
        st.metric(
            "💼 Quotes Generated",
            "847",
            delta="+23 today",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            "💰 Revenue Pipeline",
            "$2.4M",
            delta="+15.3%",
            delta_color="normal"
        )
    
    st.divider()
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 LME Price Trends (Last 30 Days)")
        
        # Generate mock historical data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        copper_base = 9500
        aluminum_base = 2650
        
        copper_prices = [copper_base + np.random.randint(-200, 300) + i*5 for i in range(30)]
        aluminum_prices = [aluminum_base + np.random.randint(-100, 150) + i*2 for i in range(30)]
        
        price_df = pd.DataFrame({
            'Date': dates,
            'Copper': copper_prices,
            'Aluminum': aluminum_prices
        })
        
        fig = px.line(
            price_df.melt(id_vars=['Date'], var_name='Metal', value_name='Price'),
            x='Date', y='Price', color='Metal',
            title="LME Prices Trend"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Business Performance")
        
        # Mock business metrics
        metrics_data = {
            'Metric': ['Quote Win Rate', 'Avg Deal Size', 'Customer Satisfaction', 'Margin %'],
            'Current': [68, 125000, 4.2, 28],
            'Target': [75, 150000, 4.5, 30],
            'Status': ['⚠️ Below', '⚠️ Below', '✅ Good', '✅ Good']
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )
    
    st.divider()
    
    # System Insights
    st.subheader("🤖 AI Insights & Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Market Intelligence**
        - Copper volatility: HIGH (15% monthly range)
        - Recommend dynamic pricing for orders >$100k
        - 3 price alerts triggered today
        """)
    
    with col2:
        st.warning("""
        **Demand Forecast**
        - Mining demand up 12% next quarter
        - Inventory alert: Reorder 540317340
        - Seasonal peak starting in 2 weeks
        """)
    
    with col3:
        st.success("""
        **Quote Optimization**
        - Bundle quotes show 23% higher win rate
        - Customer CODELCO: 85% acceptance probability
        - Pricing strategy: Competitive positioning
        """)

def show_pricing_calculator():
    """Interactive pricing calculator"""
    
    st.header("💰 Pricing Calculator")
    st.subheader("Calculate pricing using ML model + real-time costs")
    
    # Input form
    with st.form("pricing_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_id = st.selectbox(
                "Product ID",
                ["540317340", "540317341", "540317342"],
                help="Select Nexans product"
            )
            
            quantity = st.number_input(
                "Quantity (meters)",
                min_value=100,
                max_value=10000,
                value=2500,
                step=100
            )
            
            customer_segment = st.selectbox(
                "Customer Segment",
                ["mining", "industrial", "utility", "residential"]
            )
        
        with col2:
            delivery_region = st.selectbox(
                "Delivery Region",
                ["chile_central", "chile_north", "chile_south", "international"]
            )
            
            urgency = st.selectbox(
                "Urgency Level",
                ["low", "normal", "high"]
            )
            
            voltage_rating = st.number_input(
                "Voltage Rating (V)",
                min_value=1000,
                max_value=50000,
                value=5000,
                step=1000
            )
        
        calculate_btn = st.form_submit_button("🔢 Calculate Pricing", type="primary")
    
    if calculate_btn:
        with st.spinner("Calculating pricing with ML model..."):
            # Mock pricing calculation
            base_price = 45.83
            
            # Apply multipliers
            segment_mult = {"mining": 1.5, "industrial": 1.3, "utility": 1.2, "residential": 1.0}[customer_segment]
            region_mult = {"chile_north": 1.15, "chile_central": 1.0, "chile_south": 1.08, "international": 1.25}[delivery_region]
            urgency_mult = {"high": 1.15, "normal": 1.0, "low": 0.95}[urgency]
            
            final_unit_price = base_price * segment_mult * region_mult * urgency_mult
            total_price = final_unit_price * quantity
            
            # Volume discount
            if quantity > 2000:
                discount = 0.05
                total_price *= (1 - discount)
            
            # Display results
            st.success("✅ Pricing calculation completed!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Unit Price",
                    f"${final_unit_price:.2f}/meter",
                    delta=f"+{((final_unit_price/base_price-1)*100):.1f}%"
                )
            
            with col2:
                st.metric(
                    "Total Price",
                    f"${total_price:,.2f}",
                    delta=f"{quantity:,} meters"
                )
            
            with col3:
                margin = total_price * 0.25  # Estimated margin
                st.metric(
                    "Estimated Margin",
                    f"${margin:,.2f}",
                    delta="25.0%"
                )
            
            # Cost breakdown
            st.subheader("📊 Cost Breakdown")
            
            breakdown = {
                'Component': ['Material Cost', 'Manufacturing', 'Overhead', 'Margin'],
                'Amount': [total_price * 0.45, total_price * 0.20, total_price * 0.10, total_price * 0.25],
                'Percentage': ['45%', '20%', '10%', '25%']
            }
            
            breakdown_df = pd.DataFrame(breakdown)
            
            fig = px.pie(
                breakdown_df,
                values='Amount',
                names='Component',
                title="Cost Structure"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_market_intelligence():
    """Market intelligence dashboard"""
    
    st.header("📈 Market Intelligence")
    st.subheader("Real-time market monitoring and competitive analysis")
    
    # Market status
    market_data = fetch_api_data("/api/agents/market/status")
    
    if market_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Market Status",
                "🟢 Active",
                delta="Monitoring since startup"
            )
        
        with col2:
            st.metric(
                "Active Alerts",
                market_data.get("alerts_count", 0),
                delta="Real-time monitoring"
            )
        
        with col3:
            st.metric(
                "Last Update",
                "Just now",
                delta="Auto-refresh: 5min"
            )
    
    # Price volatility analysis
    st.subheader("📊 Price Volatility Analysis")
    
    # Generate mock volatility data
    dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
    volatility_data = []
    
    for i, date in enumerate(dates):
        volatility = abs(np.random.normal(0, 2)) + 1  # Base volatility
        if i > 18:  # Recent spike
            volatility += 3
        
        volatility_data.append({
            'Timestamp': date,
            'Volatility': volatility,
            'Level': 'HIGH' if volatility > 4 else 'MEDIUM' if volatility > 2 else 'LOW'
        })
    
    volatility_df = pd.DataFrame(volatility_data)
    
    fig = px.line(
        volatility_df,
        x='Timestamp',
        y='Volatility',
        color='Level',
        title="24-Hour Price Volatility"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Competitive analysis
    st.subheader("🏢 Competitive Analysis")
    
    competitive_data = {
        'Competitor': ['Prysmian Group', 'Southwire', 'General Cable', 'Nexans'],
        'Market Share': [28, 22, 18, 32],
        'Avg Price': [48.50, 43.20, 44.80, 45.83],
        'Position': ['Premium', 'Aggressive', 'Competitive', 'Competitive']
    }
    
    comp_df = pd.DataFrame(competitive_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            comp_df,
            values='Market Share',
            names='Competitor',
            title="Market Share Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            comp_df,
            x='Competitor',
            y='Avg Price',
            color='Position',
            title="Average Pricing Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Alerts and recommendations
    st.subheader("🚨 Market Alerts & Recommendations")
    
    alerts = [
        {"Type": "Price Spike", "Metal": "Copper", "Change": "+3.2%", "Action": "Review pricing strategy", "Priority": "HIGH"},
        {"Type": "Competitor Move", "Competitor": "Prysmian", "Change": "-2.1%", "Action": "Monitor closely", "Priority": "MEDIUM"},
        {"Type": "Demand Surge", "Segment": "Mining", "Change": "+12%", "Action": "Increase inventory", "Priority": "HIGH"}
    ]
    
    alerts_df = pd.DataFrame(alerts)
    st.dataframe(alerts_df, use_container_width=True, hide_index=True)

def show_demand_forecasting():
    """Demand forecasting dashboard"""
    
    st.header("🔮 Demand Forecasting")
    st.subheader("ML-powered demand predictions and inventory optimization")
    
    # Forecast parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        product_id = st.selectbox("Product", ["540317340", "540317341", "540317342"])
    
    with col2:
        forecast_days = st.selectbox("Forecast Period", [30, 60, 90])
    
    with col3:
        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner("Training ML models and generating forecast..."):
                time.sleep(2)  # Simulate processing
                st.success("✅ Forecast generated successfully!")
    
    # Mock forecast visualization
    st.subheader("📈 Demand Forecast")
    
    # Generate mock forecast data
    dates = pd.date_range(start=datetime.now(), periods=forecast_days, freq='D')
    base_demand = 1500
    
    # Historical data (past 30 days)
    hist_dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    historical_demand = []
    for i in range(30):
        seasonal = 1 + 0.2 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
        noise = np.random.normal(0, 0.1)
        demand = base_demand * seasonal * (1 + noise)
        historical_demand.append(demand)
    
    # Forecast data
    forecast_demand = []
    for i in range(forecast_days):
        seasonal = 1 + 0.2 * np.sin(2 * np.pi * i / 7)
        trend = 1 + (i * 0.001)  # Slight upward trend
        demand = base_demand * seasonal * trend
        forecast_demand.append(demand)
    
    # Combine data
    combined_dates = list(hist_dates) + list(dates)
    combined_demand = historical_demand + forecast_demand
    combined_type = ['Historical'] * 30 + ['Forecast'] * forecast_days
    
    forecast_df = pd.DataFrame({
        'Date': combined_dates,
        'Demand': combined_demand,
        'Type': combined_type
    })
    
    fig = px.line(
        forecast_df,
        x='Date',
        y='Demand',
        color='Type',
        title=f"Demand Forecast - Product {product_id}"
    )
    
    # Add confidence intervals for forecast
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[d * 1.1 for d in forecast_demand],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[d * 0.9 for d in forecast_demand],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(0,100,80,0.2)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Inventory recommendations
    st.subheader("📦 Inventory Optimization")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Stock",
            "8,450 m",
            delta="Last updated: 2h ago"
        )
    
    with col2:
        st.metric(
            "Reorder Point",
            "5,200 m",
            delta="Safety stock included"
        )
    
    with col3:
        st.metric(
            "EOQ",
            "12,000 m",
            delta="Optimal order quantity"
        )
    
    with col4:
        st.metric(
            "Days of Stock",
            "5.6 days",
            delta="At current demand rate",
            delta_color="inverse"
        )
    
    # Alerts
    st.warning("⚠️ **Inventory Alert**: Stock level approaching reorder point. Recommend placing order within 2 days.")
    st.info("💡 **Optimization Tip**: Seasonal demand peak expected in 2 weeks. Consider 15% buffer stock.")

def show_quote_generator():
    """Quote generation interface"""
    
    st.header("📄 Quote Generator")
    st.subheader("AI-powered automated quote generation")
    
    # Quote form
    with st.form("quote_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Customer Information**")
            customer_id = st.text_input("Customer ID", value="CODELCO_001")
            customer_segment = st.selectbox(
                "Customer Segment",
                ["mining", "industrial", "utility", "residential"]
            )
            
            st.markdown("**Project Details**")
            project_name = st.text_input("Project Name", value="Mine Expansion Phase 2")
            budget_limit = st.number_input("Budget Limit (USD)", min_value=0, value=200000, step=10000)
        
        with col2:
            st.markdown("**Product Requirements**")
            
            # Dynamic product addition
            if 'quote_products' not in st.session_state:
                st.session_state.quote_products = [
                    {"product_id": "540317340", "quantity": 2500, "requirements": ["fire_resistant"]}
                ]
            
            for i, product in enumerate(st.session_state.quote_products):
                st.write(f"**Product {i+1}**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    product["product_id"] = st.selectbox(
                        f"Product ID {i+1}",
                        ["540317340", "540317341", "540317342"],
                        key=f"product_{i}"
                    )
                
                with col_b:
                    product["quantity"] = st.number_input(
                        f"Quantity {i+1} (m)",
                        min_value=100,
                        value=product["quantity"],
                        key=f"qty_{i}"
                    )
            
            delivery_location = st.selectbox(
                "Delivery Location",
                ["chile_central", "chile_north", "chile_south", "international"]
            )
            
            delivery_deadline = st.date_input(
                "Delivery Deadline",
                value=datetime.now() + timedelta(days=60)
            )
        
        generate_quote_btn = st.form_submit_button("🎯 Generate Quote", type="primary")
    
    if generate_quote_btn:
        with st.spinner("Generating optimized quote with AI..."):
            time.sleep(3)  # Simulate processing
            
            # Mock quote generation
            quote_id = f"AQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate totals
            total_quantity = sum(p["quantity"] for p in st.session_state.quote_products)
            base_price = 45.83
            
            # Apply business rules
            segment_mult = {"mining": 1.5, "industrial": 1.3, "utility": 1.2, "residential": 1.0}[customer_segment]
            region_mult = {"chile_north": 1.15, "chile_central": 1.0, "chile_south": 1.08, "international": 1.25}[delivery_location]
            
            unit_price = base_price * segment_mult * region_mult
            subtotal = unit_price * total_quantity
            
            # Volume discount
            if total_quantity > 2000:
                discount_rate = 0.05
                discount = subtotal * discount_rate
                subtotal -= discount
            else:
                discount = 0
            
            taxes = subtotal * 0.19  # 19% IVA
            total_price = subtotal + taxes
            
            st.success(f"✅ Quote {quote_id} generated successfully!")
            
            # Quote summary
            st.subheader("📋 Quote Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Quote ID", quote_id)
                st.metric("Customer", customer_id)
                st.metric("Total Quantity", f"{total_quantity:,} m")
            
            with col2:
                st.metric("Unit Price", f"${unit_price:.2f}/m")
                st.metric("Subtotal", f"${subtotal:,.2f}")
                if discount > 0:
                    st.metric("Volume Discount", f"-${discount:,.2f}")
            
            with col3:
                st.metric("Taxes (19%)", f"${taxes:,.2f}")
                st.metric("**Total Price**", f"**${total_price:,.2f}**")
                margin = total_price * 0.25
                st.metric("Est. Margin", f"${margin:,.2f}")
            
            # Quote details table
            st.subheader("📊 Quote Line Items")
            
            line_items = []
            for i, product in enumerate(st.session_state.quote_products):
                line_total = product["quantity"] * unit_price
                line_items.append({
                    "Item": i+1,
                    "Product ID": product["product_id"],
                    "Description": f"Nexans Cable {product['product_id']}",
                    "Quantity": f"{product['quantity']:,} m",
                    "Unit Price": f"${unit_price:.2f}",
                    "Line Total": f"${line_total:,.2f}"
                })
            
            quote_df = pd.DataFrame(line_items)
            st.dataframe(quote_df, use_container_width=True, hide_index=True)
            
            # AI insights
            st.subheader("🤖 AI Quote Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Customer Analysis**
                - Segment: {customer_segment.title()}
                - Historical win rate: 73%
                - Avg acceptance time: 4.2 days
                - Price sensitivity: Medium
                """)
            
            with col2:
                st.success(f"""
                **Optimization Applied**
                - Volume discount: {discount_rate*100 if discount > 0 else 0:.0f}%
                - Regional adjustment: {(region_mult-1)*100:+.0f}%
                - Segment premium: {(segment_mult-1)*100:+.0f}%
                - Win probability: 78%
                """)
            
            # Download quote
            st.download_button(
                "📥 Download Quote PDF",
                data="Mock PDF content - In production would generate actual PDF",
                file_name=f"Quote_{quote_id}.pdf",
                mime="application/pdf"
            )

def show_system_monitor():
    """System monitoring dashboard"""
    
    st.header("⚙️ System Monitor")
    st.subheader("Real-time system performance and health metrics")
    
    # System status
    system_data = fetch_system_status()
    health_data = fetch_health_check()
    
    if system_data and health_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "System Status",
                "🟢 Healthy" if health_data.get("status") == "healthy" else "🔴 Issues",
                delta=f"Uptime: {system_data['system_info']['uptime_hours']:.1f}h"
            )
        
        with col2:
            st.metric(
                "API Requests",
                f"{system_data['performance_metrics']['total_requests']:,}",
                delta="All time"
            )
        
        with col3:
            st.metric(
                "Avg Response",
                system_data['performance_metrics']['average_response_time'],
                delta="Target: <200ms"
            )
        
        with col4:
            st.metric(
                "Error Rate",
                system_data['performance_metrics']['error_rate'],
                delta="Target: <1%"
            )
    
    # Agent status
    st.subheader("🤖 Intelligent Agents Status")
    
    if system_data:
        agents = system_data['agent_status']
        
        for agent_name, agent_info in agents.items():
            with st.expander(f"📊 {agent_name.replace('_', ' ').title()} Agent"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Status:** {agent_info['status'].title()}")
                    if 'monitoring_active' in agent_info:
                        st.write(f"**Monitoring:** {'✅ Active' if agent_info['monitoring_active'] else '❌ Inactive'}")
                    if 'models_available' in agent_info:
                        st.write(f"**Models:** {', '.join(agent_info['models_available'])}")
                
                with col2:
                    st.write("**Capabilities:**")
                    for capability in agent_info['capabilities']:
                        st.write(f"• {capability}")
    
    # Performance charts
    st.subheader("📈 Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time chart
        times = pd.date_range(end=datetime.now(), periods=24, freq='H')
        response_times = [150 + np.random.randint(-30, 50) for _ in range(24)]
        
        perf_df = pd.DataFrame({
            'Time': times,
            'Response Time (ms)': response_times
        })
        
        fig = px.line(
            perf_df,
            x='Time',
            y='Response Time (ms)',
            title="24-Hour Response Time"
        )
        fig.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Target: 200ms")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Request volume chart
        request_counts = [np.random.randint(50, 200) for _ in range(24)]
        
        vol_df = pd.DataFrame({
            'Time': times,
            'Requests': request_counts
        })
        
        fig = px.bar(
            vol_df,
            x='Time',
            y='Requests',
            title="24-Hour Request Volume"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # System logs
    st.subheader("📝 Recent System Logs")
    
    logs = [
        {"Time": "2024-01-15 14:23:15", "Level": "INFO", "Message": "Market Intelligence Agent: Price alert triggered for Copper"},
        {"Time": "2024-01-15 14:22:03", "Level": "INFO", "Message": "Quote generated successfully: AQ_20240115_142203"},
        {"Time": "2024-01-15 14:21:45", "Level": "INFO", "Message": "Demand forecast completed for product 540317340"},
        {"Time": "2024-01-15 14:20:12", "Level": "WARN", "Message": "LME API response time exceeded 5 seconds"},
        {"Time": "2024-01-15 14:19:33", "Level": "INFO", "Message": "Health check completed - all systems operational"}
    ]
    
    logs_df = pd.DataFrame(logs)
    st.dataframe(logs_df, use_container_width=True, hide_index=True)

# Run the app
if __name__ == "__main__":
    main()