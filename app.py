import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration to wide layout and set title and favicon
st.set_page_config(
    page_title="Tech Layoffs & Hiring Trends Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and premium styling (custom card components, gradient header, etc.)
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Elegant Title Gradient */
    .title-gradient {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Styled Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .metric-delta.up {
        color: #10b981;
    }
    
    .metric-delta.down {
        color: #ef4444;
    }
    
    /* Footer style */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #64748b;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING (WITH CACHING)
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    """
    Loads tech layoffs dataset and runs the preprocessing logic
    reused from the notebook: filling missing values for numerical,
    categorical, and discrete numerical features.
    """
    try:
        # Load dataset from current directory
        df = pd.read_csv('tech_layoffs_hiring_trends_0.csv')
        
        # Numerical Columns (Real Numerical)
        num_cols = ['layoffs_count', 'open_roles']
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
                
        # Categorical and Discrete Numerical Columns
        cat_cols = [
            'company_name', 'industry', 'country', 'company_size',
            'layoff_percentage', 'reason_for_layoffs', 'ai_automation_impact',
            'ai_replacement_risk', 'hiring_trend', 'remote_jobs_percentage',
            'top_hiring_role', 'stock_growth_percent', 'revenue_growth_percent',
            'salary_budget_change', 'ai_adoption_level', 'employee_sentiment',
            'job_security_score', 'market_condition'
        ]
        for col in cat_cols:
            if col in df.columns:
                modes = df[col].mode()
                if not modes.empty:
                    df[col] = df[col].fillna(modes[0])
                else:
                    df[col] = df[col].fillna("Unknown")
                    
        # Ensure year and month columns are formatted
        if 'year' in df.columns:
            df['year'] = df['year'].astype(int)
            
        return df
    except FileNotFoundError:
        st.error("Dataset file 'tech_layoffs_hiring_trends_0.csv' not found. Please place it in the same folder as app.py.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading and preprocessing data: {e}")
        return pd.DataFrame()

# Load the data
df_original = load_and_preprocess_data()

if df_original.empty:
    st.stop()

# Copy original data for operations
df = df_original.copy()

# ---------------------------------------------------------
# 2. SIDEBAR - METRICS OVERVIEW & FILTERS
# ---------------------------------------------------------
st.sidebar.markdown("<h2 style='text-align: center; color: #3b82f6; font-weight: 800;'>Tech Layoffs EDA</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation menu
st.sidebar.markdown("### 🗺️ Navigation")
page = st.sidebar.radio(
    "Go to page:",
    [
        "📊 Overview & Metrics",
        "🛑 Layoffs & Demographics",
        "💼 Hiring Trends & Roles",
        "🤖 AI Impact & Job Security",
        "🧮 Advanced & Custom Analysis"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filter Controls")

# Filtering option - Expander for clean UI
with st.sidebar.expander("Apply Filters", expanded=True):
    # Year filter
    years = sorted(df['year'].unique()) if 'year' in df.columns else []
    selected_years = st.multiselect("Select Year(s):", options=years, default=years)
    
    # Country filter
    countries = sorted(df['country'].unique()) if 'country' in df.columns else []
    selected_countries = st.multiselect("Select Country/Countries:", options=countries, default=countries)
    
    # Industry filter
    industries = sorted(df['industry'].unique()) if 'industry' in df.columns else []
    selected_industries = st.multiselect("Select Industry:", options=industries, default=industries)
    
    # Company Size filter
    sizes = sorted(df['company_size'].unique()) if 'company_size' in df.columns else []
    selected_sizes = st.multiselect("Select Company Size:", options=sizes, default=sizes)
    
    # Market Condition filter
    markets = sorted(df['market_condition'].unique()) if 'market_condition' in df.columns else []
    selected_markets = st.multiselect("Select Market Condition:", options=markets, default=markets)

# Filter the DataFrame based on inputs
df_filtered = df.copy()

if selected_years:
    df_filtered = df_filtered[df_filtered['year'].isin(selected_years)]
if selected_countries:
    df_filtered = df_filtered[df_filtered['country'].isin(selected_countries)]
if selected_industries:
    df_filtered = df_filtered[df_filtered['industry'].isin(selected_industries)]
if selected_sizes:
    df_filtered = df_filtered[df_filtered['company_size'].isin(selected_sizes)]
if selected_markets:
    df_filtered = df_filtered[df_filtered['market_condition'].isin(selected_markets)]

# Sidebar Dataset Overview Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Overview")
st.sidebar.markdown(f"**Total Records:** {df_original.shape[0]:,}")
st.sidebar.markdown(f"**Filtered Records:** {df_filtered.shape[0]:,}")
st.sidebar.markdown(f"**Features (Columns):** {df_original.shape[1]}")

# ---------------------------------------------------------
# Helper Functions for Visualizations
# ---------------------------------------------------------
def create_metric_card(title, value, delta=None, delta_direction="up"):
    """
    Renders a stylized glassmorphism metric box
    """
    delta_class = "up" if delta_direction == "up" else "down"
    delta_symbol = "▲" if delta_direction == "up" else "▼"
    delta_html = f"<div class='metric-delta {delta_class}'>{delta_symbol} {delta}</div>" if delta else ""
    
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>{title}</div>
        <div class='metric-value'>{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 1: OVERVIEW & KEY METRICS
# ---------------------------------------------------------
if page == "📊 Overview & Metrics":
    st.markdown("<h1 class='title-gradient'>Tech Layoffs & Hiring Trends Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>A clean, interactive dashboard analyzing layoffs, hiring dynamics, and AI implications in the tech workforce.</p>", unsafe_allow_html=True)
    
    # Verify dataset has required columns before loading metrics
    try:
        total_layoffs = df_filtered['layoffs_count'].sum() if 'layoffs_count' in df_filtered.columns else 0
        total_open_roles = df_filtered['open_roles'].sum() if 'open_roles' in df_filtered.columns else 0
        avg_layoff_pct = df_filtered['layoff_percentage'].mean() if 'layoff_percentage' in df_filtered.columns else 0.0
        avg_job_security = df_filtered['job_security_score'].mean() if 'job_security_score' in df_filtered.columns else 0.0
        
        # Calculate comparison for delta (e.g. comparing to unfiltered average)
        orig_avg_security = df_original['job_security_score'].mean() if 'job_security_score' in df_original.columns else 0.0
        security_diff = avg_job_security - orig_avg_security
        security_delta_dir = "up" if security_diff >= 0 else "down"
        security_delta_text = f"{abs(security_diff):.2f} vs overall avg"
        
        # Display custom metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            create_metric_card("Total Layoffs Count", f"{total_layoffs:,.0f}")
        with m2:
            create_metric_card("Total Open Roles Available", f"{total_open_roles:,.0f}")
        with m3:
            create_metric_card("Average Layoff Percentage", f"{avg_layoff_pct:.1f}%")
        with m4:
            create_metric_card("Average Job Security Score", f"{avg_job_security:.2f}/10", security_delta_text, security_delta_dir)
            
    except Exception as e:
        st.warning(f"Unable to calculate some KPI metrics: {e}")
        
    st.markdown("### 📋 Interactive Data Preview")
    
    tab1, tab2 = st.tabs(["📊 Filtered Data Explorer", "📈 Summary Statistics"])
    
    with tab1:
        st.markdown("Browse and search the filtered dataset. Use column headers to sort.")
        st.dataframe(df_filtered, use_container_width=True, height=350)
        
    with tab2:
        st.markdown("Statistical summary of numerical values in the filtered dataset.")
        st.dataframe(df_filtered.describe(), use_container_width=True)
        
    st.markdown("### 📊 Distribution of Key Metric Columns")
    c1, c2 = st.columns(2)
    
    with c1:
        if 'layoffs_count' in df_filtered.columns:
            fig1 = px.histogram(
                df_filtered, 
                x='layoffs_count', 
                nbins=30, 
                marginal='box',
                title='Distribution of Layoffs count per Company',
                color_discrete_sequence=['#ef4444'], # Red-coral accent
                template='plotly_dark'
            )
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', title_font_size=16)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("layoffs_count column not available.")
            
    with c2:
        if 'open_roles' in df_filtered.columns:
            fig2 = px.histogram(
                df_filtered, 
                x='open_roles', 
                nbins=30, 
                marginal='box',
                title='Distribution of Open Roles per Company',
                color_discrete_sequence=['#10b981'], # Emerald accent
                template='plotly_dark'
            )
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', title_font_size=16)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("open_roles column not available.")

# ---------------------------------------------------------
# PAGE 2: LAYOFFS & DEMOGRAPHICS
# ---------------------------------------------------------
elif page == "🛑 Layoffs & Demographics":
    st.markdown("<h2 class='title-gradient'>Workforce Layoffs & Demographics</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Explore details on the companies, countries, industries, and reasons for tech layoffs.</p>", unsafe_allow_html=True)
    
    agg_type = st.radio("Select Bar Chart Aggregation Type:", ["Total Layoffs (Sum)", "Average Layoffs per Record (Mean)", "Number of Layoff Events (Count)"], horizontal=True)
    
    # Map aggregation type to aggregation functions
    if agg_type == "Total Layoffs (Sum)":
        y_val = "layoffs_count"
        agg_func = "sum"
        label_y = "Total Layoffs Count"
    elif agg_type == "Average Layoffs per Record (Mean)":
        y_val = "layoffs_count"
        agg_func = "mean"
        label_y = "Average Layoffs Count"
    else:
        y_val = "layoffs_count"
        agg_func = "count"
        label_y = "Number of Layoff Events"
        
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 🏢 Top Affected Companies")
        if 'company_name' in df_filtered.columns and 'layoffs_count' in df_filtered.columns:
            # Aggregate data
            comp_data = df_filtered.groupby('company_name')[y_val].agg(agg_func).reset_index()
            comp_data = comp_data.sort_values(by=y_val, ascending=False).head(15)
            
            fig_comp = px.bar(
                comp_data,
                x='company_name',
                y=y_val,
                title=f'Top 15 Companies by {label_y}',
                color=y_val,
                color_continuous_scale='Reds',
                labels={'company_name': 'Company Name', y_val: label_y},
                template='plotly_dark'
            )
            fig_comp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("Required columns ('company_name', 'layoffs_count') are missing.")
            
    with c2:
        st.markdown("#### 🌐 Countries with Highest Layoffs")
        if 'country' in df_filtered.columns and 'layoffs_count' in df_filtered.columns:
            country_data = df_filtered.groupby('country')[y_val].agg(agg_func).reset_index()
            country_data = country_data.sort_values(by=y_val, ascending=False).head(15)
            
            fig_country = px.bar(
                country_data,
                x='country',
                y=y_val,
                title=f'Top Countries by {label_y}',
                color=y_val,
                color_continuous_scale='Oranges',
                labels={'country': 'Country', y_val: label_y},
                template='plotly_dark'
            )
            fig_country.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
            st.plotly_chart(fig_country, use_container_width=True)
        else:
            st.warning("Required columns ('country', 'layoffs_count') are missing.")
            
    # Industry & Reason for Layoffs
    st.markdown("---")
    c3, c4 = st.columns([3, 2])
    
    with c3:
        st.markdown("#### 🏢 Industry vs Layoffs")
        if 'industry' in df_filtered.columns and 'layoffs_count' in df_filtered.columns:
            ind_data = df_filtered.groupby('industry')[y_val].agg(agg_func).reset_index()
            ind_data = ind_data.sort_values(by=y_val, ascending=False)
            
            fig_ind = px.bar(
                ind_data,
                x='industry',
                y=y_val,
                title=f'Layoffs by Industry sector ({label_y})',
                color='industry',
                color_discrete_sequence=px.colors.qualitative.Alphabet,
                labels={'industry': 'Industry Sector', y_val: label_y},
                template='plotly_dark'
            )
            fig_ind.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_ind, use_container_width=True)
        else:
            st.warning("Required columns missing.")
            
    with c4:
        st.markdown("#### ❓ Primary Reasons for Layoffs")
        if 'reason_for_layoffs' in df_filtered.columns:
            reasons_count = df_filtered['reason_for_layoffs'].value_counts().reset_index()
            reasons_count.columns = ['Reason', 'Count']
            
            fig_reasons = px.pie(
                reasons_count,
                values='Count',
                names='Reason',
                title='Reason Share for Tech Layoffs',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template='plotly_dark'
            )
            fig_reasons.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_reasons, use_container_width=True)
        else:
            st.warning("reason_for_layoffs column is missing.")

    # Layoffs over time and country comparisons
    st.markdown("---")
    st.markdown("### ⏱️ Temporal & Location Intersect Analyses")
    
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("#### 🗓️ Layoffs Distribution Over Time (Yearly)")
        if 'year' in df_filtered.columns and 'layoffs_count' in df_filtered.columns:
            # We can show a scatterplot (from notebook) and an aggregated trendline
            sub_tab1, sub_tab2 = st.tabs(["Aggregated Box Plot (Recommended)", "Scatter Plot (Original)"])
            with sub_tab1:
                fig_time_box = px.box(
                    df_filtered,
                    x='year',
                    y='layoffs_count',
                    points="all",
                    title='Layoffs Distribution by Year',
                    color='year',
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    template='plotly_dark'
                )
                fig_time_box.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig_time_box, use_container_width=True)
            with sub_tab2:
                fig_time_scat = px.scatter(
                    df_filtered,
                    x='year',
                    y='layoffs_count',
                    opacity=0.6,
                    title='Layoffs Scatter Plot Over Years',
                    color_discrete_sequence=['#fbbf24'],
                    template='plotly_dark'
                )
                fig_time_scat.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_time_scat, use_container_width=True)
        else:
            st.warning("Year/Layoffs count missing.")
            
    with c6:
        st.markdown("#### 🗺️ Country-wise Reasons for Layoffs")
        if 'country' in df_filtered.columns and 'reason_for_layoffs' in df_filtered.columns:
            fig_country_reasons = px.histogram(
                df_filtered,
                x='country',
                color='reason_for_layoffs',
                barmode='group',
                title='Layoff Reasons Across Different Countries',
                labels={'country': 'Country', 'count': 'Events Count'},
                color_discrete_sequence=px.colors.qualitative.Vivid,
                template='plotly_dark'
            )
            fig_country_reasons.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_country_reasons, use_container_width=True)
        else:
            st.warning("Required columns missing.")

# ---------------------------------------------------------
# PAGE 3: HIRING TRENDS & ROLES
# ---------------------------------------------------------
elif page == "💼 Hiring Trends & Roles":
    st.markdown("<h2 class='title-gradient'>Hiring Trends & Demanded Roles</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Analyze hiring trajectories and find the most sought-after jobs in tech companies.</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📈 Overall Hiring Trend Share")
        if 'hiring_trend' in df_filtered.columns:
            ht_counts = df_filtered['hiring_trend'].value_counts().reset_index()
            ht_counts.columns = ['Hiring Trend', 'Count']
            
            fig_ht = px.pie(
                ht_counts,
                values='Count',
                names='Hiring Trend',
                title='Current Corporate Hiring Trajectory',
                hole=0.4,
                color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'], # custom modern colors
                template='plotly_dark'
            )
            fig_ht.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_ht, use_container_width=True)
        else:
            st.warning("hiring_trend column not found.")
            
    with c2:
        st.markdown("#### 🎯 Top In-Demand Hiring Roles")
        if 'top_hiring_role' in df_filtered.columns:
            roles_count = df_filtered['top_hiring_role'].value_counts().reset_index()
            roles_count.columns = ['Role', 'Count']
            
            fig_roles = px.pie(
                roles_count,
                values='Count',
                names='Role',
                title='Most In-Demand Workforce Roles',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Safe,
                template='plotly_dark'
            )
            fig_roles.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_roles, use_container_width=True)
        else:
            st.warning("top_hiring_role column not found.")
            
    st.markdown("---")
    st.markdown("### 🏢 Hiring Distribution Across Corporate Segments")
    
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 🏢 Hiring Trends by Company Size")
        if 'company_size' in df_filtered.columns and 'hiring_trend' in df_filtered.columns:
            fig_sz_ht = px.histogram(
                df_filtered,
                x='company_size',
                color='hiring_trend',
                barmode='group',
                title='Hiring Trends Breakdown across Company Scale',
                color_discrete_sequence=px.colors.qualitative.Set2,
                template='plotly_dark'
            )
            fig_sz_ht.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sz_ht, use_container_width=True)
        else:
            st.warning("Required columns missing.")
            
    with c4:
        st.markdown("#### 🚀 Top Hiring Roles by Company Scale")
        if 'company_size' in df_filtered.columns and 'top_hiring_role' in df_filtered.columns:
            # Crosstab visualization from notebook
            ct_df = pd.crosstab(df_filtered['company_size'], df_filtered['top_hiring_role']).reset_index()
            ct_melted = ct_df.melt(id_vars='company_size', value_vars=ct_df.columns[1:], var_name='Role', value_name='Count')
            
            fig_ct = px.bar(
                ct_melted,
                x='company_size',
                y='Count',
                color='Role',
                barmode='group',
                title='In-demand Job Roles grouped by Company Size',
                color_discrete_sequence=px.colors.qualitative.Alphabet,
                template='plotly_dark'
            )
            fig_ct.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_ct, use_container_width=True)
        else:
            st.warning("Required columns missing.")
            
    # Hiring trends by Industry
    st.markdown("---")
    st.markdown("#### 🏭 Industry hiring trends")
    if 'industry' in df_filtered.columns and 'hiring_trend' in df_filtered.columns:
        fig_ind_ht = px.histogram(
            df_filtered,
            x='industry',
            color='hiring_trend',
            barmode='group',
            title='Hiring Trajectory across Industry Sectors',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template='plotly_dark'
        )
        fig_ind_ht.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_ind_ht, use_container_width=True)

# ---------------------------------------------------------
# PAGE 4: AI IMPACT & JOB SECURITY
# ---------------------------------------------------------
elif page == "🤖 AI Impact & Job Security":
    st.markdown("<h2 class='title-gradient'>AI Impact, Replacement Risks & Job Security</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Analyzing the role of AI automation, potential job replacement risks, and overall stability metrics.</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 🔐 Total Job Security Sum by Size & Country")
        if 'company_size' in df_filtered.columns and 'country' in df_filtered.columns and 'job_security_score' in df_filtered.columns:
            # Pivot table from notebook
            pivot_js = pd.pivot_table(
                df_filtered,
                columns='country',
                index='company_size',
                values='job_security_score',
                aggfunc='sum'
            ).reset_index()
            
            # Melt for Plotly
            pivot_melted = pivot_js.melt(id_vars='company_size', value_vars=pivot_js.columns[1:], var_name='Country', value_name='Job Security Score')
            
            fig_js = px.bar(
                pivot_melted,
                x='company_size',
                y='Job Security Score',
                color='Country',
                barmode='group',
                title='Country-wise Job Security Sum by Company Size',
                color_discrete_sequence=px.colors.qualitative.Bold,
                template='plotly_dark'
            )
            fig_js.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_js, use_container_width=True)
        else:
            st.warning("Required columns ('company_size', 'country', 'job_security_score') missing.")
            
    with c2:
        st.markdown("#### 🤖 AI Replacement Risk by Scale & Company")
        if 'company_name' in df_filtered.columns and 'company_size' in df_filtered.columns and 'ai_replacement_risk' in df_filtered.columns:
            pivot_ai = pd.pivot_table(
                df_filtered,
                columns='company_size',
                index='company_name',
                values='ai_replacement_risk',
                aggfunc='sum'
            ).reset_index()
            
            pivot_ai_melt = pivot_ai.melt(id_vars='company_name', value_vars=pivot_ai.columns[1:], var_name='Company Size', value_name='AI Replacement Risk Sum')
            
            fig_ai = px.bar(
                pivot_ai_melt,
                x='company_name',
                y='AI Replacement Risk Sum',
                color='Company Size',
                title='AI Replacement Risk Sum Across Companies',
                template='plotly_dark'
            )
            fig_ai.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_ai, use_container_width=True)
        else:
            st.warning("Required columns missing.")
            
    st.markdown("---")
    st.markdown("### 🗺️ Reason Matrix & Automation Intersect")
    c3, c4 = st.columns(2)
    
    with c3:
        st.markdown("#### 🗓️ Layoffs Reason Matrix Across Market Conditions")
        if 'market_condition' in df_filtered.columns and 'reason_for_layoffs' in df_filtered.columns and 'layoffs_count' in df_filtered.columns:
            pivot_reason = pd.pivot_table(
                df_filtered,
                columns='reason_for_layoffs',
                index='market_condition',
                values='layoffs_count',
                aggfunc='count'
            )
            
            fig_heat = px.imshow(
                pivot_reason,
                labels=dict(x="Reason for Layoffs", y="Market Condition", color="Layoff Count"),
                x=pivot_reason.columns,
                y=pivot_reason.index,
                color_continuous_scale='YlOrRd',
                title='Layoffs Reasons Across Market Conditions Heatmap',
                text_auto=True,
                template='plotly_dark'
            )
            fig_heat.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("Required columns missing.")
            
    with c4:
        st.markdown("#### 🏢 Reasons for Layoffs vs Company Size")
        if 'company_size' in df_filtered.columns and 'reason_for_layoffs' in df_filtered.columns:
            fig_sz_reason = px.histogram(
                df_filtered,
                x='company_size',
                color='reason_for_layoffs',
                barmode='group',
                title='Layoff Reasons distributed by Corporate Scale',
                color_discrete_sequence=px.colors.qualitative.Prism,
                template='plotly_dark'
            )
            fig_sz_reason.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sz_reason, use_container_width=True)
        else:
            st.warning("Required columns missing.")

# ---------------------------------------------------------
# PAGE 5: ADVANCED & CUSTOM ANALYSIS
# ---------------------------------------------------------
elif page == "🧮 Advanced & Custom Analysis":
    st.markdown("<h2 class='title-gradient'>Advanced Analytics & Custom Query Builder</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Examine correlations across features or build your own custom plotly charts adaptively.</p>", unsafe_allow_html=True)
    
    tab_corr, tab_custom = st.tabs(["🔗 Correlation Matrix", "🛠️ Custom Interactive Plot Builder"])
    
    with tab_corr:
        st.markdown("#### 🔗 Pearson Correlation of Numeric Attributes")
        numeric_df = df_filtered.select_dtypes(include=['int64', 'float64', 'int32'])
        
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr = numeric_df.corr()
            
            fig_corr = px.imshow(
                corr,
                labels=dict(color="Correlation Coefficient"),
                x=corr.columns,
                y=corr.index,
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1],
                title='Correlation Heatmap',
                text_auto='.2f',
                template='plotly_dark'
            )
            fig_corr.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Not enough numeric columns available for correlation.")
            
    with tab_custom:
        st.markdown("#### 🛠️ Dynamically Adapt Charts to Columns")
        st.write("Construct custom plots that automatically adapt to your dataset's columns.")
        
        # User selections
        all_cols = list(df_filtered.columns)
        num_cols = list(df_filtered.select_dtypes(include=['int64', 'float64', 'int32']).columns)
        cat_cols = list(df_filtered.select_dtypes(include=['object']).columns)
        
        c_layout1, c_layout2, c_layout3 = st.columns(3)
        with c_layout1:
            chart_type = st.selectbox("Select Chart Type:", ["Bar Chart", "Scatter Plot", "Line Chart", "Box Plot", "Pie Chart", "Histogram"])
        with c_layout2:
            x_col = st.selectbox("Select X-Axis/Label Column:", all_cols, index=all_cols.index('industry') if 'industry' in all_cols else 0)
        with c_layout3:
            # Y value only needed for certain chart types
            y_col_options = ["None"] + num_cols
            y_col = st.selectbox("Select Y-Axis/Value Column:", y_col_options, index=y_col_options.index('layoffs_count') if 'layoffs_count' in y_col_options else 0)
            
        color_col_options = ["None"] + cat_cols + ['year']
        color_col = st.selectbox("Color/Group By Column:", color_col_options, index=0)
        
        # Build and render the custom Plotly chart
        try:
            custom_fig = None
            color_arg = None if color_col == "None" else color_col
            
            if chart_type == "Bar Chart":
                if y_col == "None":
                    st.error("Please select a Y-Axis/Value column for a Bar Chart.")
                else:
                    agg_option = st.selectbox("Aggregation function for Y:", ["Sum", "Mean", "Count"])
                    if agg_option == "Sum":
                        custom_df = df_filtered.groupby([x_col] + ([color_col] if color_col != "None" else [])).sum(numeric_only=True).reset_index()
                    elif agg_option == "Mean":
                        custom_df = df_filtered.groupby([x_col] + ([color_col] if color_col != "None" else [])).mean(numeric_only=True).reset_index()
                    else:
                        custom_df = df_filtered.groupby([x_col] + ([color_col] if color_col != "None" else [])).count().reset_index()
                        
                    custom_fig = px.bar(
                        custom_df,
                        x=x_col,
                        y=y_col if agg_option != "Count" else custom_df.columns[-1],
                        color=color_arg,
                        title=f"{agg_option} of {y_col} by {x_col}",
                        template='plotly_dark',
                        barmode='group'
                    )
                    
            elif chart_type == "Scatter Plot":
                if y_col == "None":
                    st.error("Please select a Y-Axis/Value column for a Scatter Plot.")
                else:
                    custom_fig = px.scatter(
                        df_filtered,
                        x=x_col,
                        y=y_col,
                        color=color_arg,
                        title=f"Scatter Plot of {y_col} vs {x_col}",
                        template='plotly_dark'
                    )
                    
            elif chart_type == "Line Chart":
                if y_col == "None":
                    st.error("Please select a Y-Axis/Value column for a Line Chart.")
                else:
                    custom_df = df_filtered.groupby([x_col] + ([color_col] if color_col != "None" else [])).mean(numeric_only=True).reset_index()
                    custom_fig = px.line(
                        custom_df,
                        x=x_col,
                        y=y_col,
                        color=color_arg,
                        title=f"Line Chart (Mean) of {y_col} over {x_col}",
                        template='plotly_dark'
                    )
                    
            elif chart_type == "Box Plot":
                if y_col == "None":
                    st.error("Please select a Y-Axis/Value column for a Box Plot.")
                else:
                    custom_fig = px.box(
                        df_filtered,
                        x=x_col,
                        y=y_col,
                        color=color_arg,
                        title=f"Box Plot of {y_col} distribution by {x_col}",
                        template='plotly_dark'
                    )
                    
            elif chart_type == "Pie Chart":
                pie_counts = df_filtered[x_col].value_counts().reset_index()
                pie_counts.columns = [x_col, 'Count']
                custom_fig = px.pie(
                    pie_counts,
                    values='Count',
                    names=x_col,
                    title=f"Distribution of {x_col}",
                    hole=0.4,
                    template='plotly_dark'
                )
                
            elif chart_type == "Histogram":
                custom_fig = px.histogram(
                    df_filtered,
                    x=x_col,
                    color=color_arg,
                    title=f"Histogram distribution of {x_col}",
                    template='plotly_dark',
                    barmode='group'
                )
                
            if custom_fig:
                custom_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(custom_fig, use_container_width=True)
                
        except Exception as err:
            st.error(f"Error rendering custom chart: {err}. Try picking different columns.")

# ---------------------------------------------------------
# FOOTER - PREMIUM TOUCH
# ---------------------------------------------------------
st.markdown("""
<div class='footer'>
    Tech Layoffs & Hiring Trends Analysis Dashboard • Reusing Colab Analysis Logic • Designed with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
