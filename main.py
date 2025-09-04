import calendar
import io
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="LLM Training Payment Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.insight-box {
    background: #1e293b;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #4f46e5;
    margin: 1rem 0;
    border: 1px solid #475569;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    color: #e2e8f0;
}

.warning-box {
    background: #1e293b;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #f59e0b;
    margin: 1rem 0;
    border: 1px solid #475569;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    color: #e2e8f0;
}

.success-box {
    background: #1e293b;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #10b981;
    margin: 1rem 0;
    border: 1px solid #475569;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    color: #e2e8f0;
}

.debug-info {
    background: #1e293b;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #475569;
    margin: 1rem 0;
    font-family: monospace;
    font-size: 0.9em;
    color: #e2e8f0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}
</style>
</style>
""", unsafe_allow_html=True)

class PaymentAnalytics:
    def __init__(self):
        self.quality_multipliers = {
            'approved perfectly': 1.2,
            'approved with minor edits': 1.0,
            'approved with major fixes': 0.8,
            'waiting for review': 0.0,
            'rejected': 0.0,
            'unusable': 0.0
        }
    
    def parse_time_string(self, time_str: str) -> float:
        """Parse time strings like '1h 19m', '34 minutes', '3 hours' into decimal hours"""
        if not time_str or pd.isna(time_str):
            return 0.0
            
        time_str = str(time_str).lower().strip()
        
        # Remove "View Task" if present
        time_str = re.sub(r'\s*view\s*task\s*$', '', time_str)
        
        hours = 0.0
        minutes = 0.0
        
        # Extract hours (e.g., "1h", "3.5h")
        hour_match = re.search(r'(\d+(?:\.\d+)?)\s*h', time_str)
        if hour_match:
            hours = float(hour_match.group(1))
        
        # Extract minutes (e.g., "19m", "34 minutes")
        minute_match = re.search(r'(\d+)\s*m(?:in)?', time_str)
        if minute_match:
            minutes = float(minute_match.group(1))
        
        # Handle "X hours" format
        if 'hour' in time_str and not hour_match:
            hour_only_match = re.search(r'(\d+(?:\.\d+)?)', time_str)
            if hour_only_match:
                hours = float(hour_only_match.group(1))
        
        # Handle "X minutes" format
        if ('minute' in time_str or 'min' in time_str) and not 'hour' in time_str and not minute_match:
            minute_only_match = re.search(r'(\d+)', time_str)
            if minute_only_match:
                minutes = float(minute_only_match.group(1))
        
        return hours + (minutes / 60.0)
    
    def get_quality_multiplier(self, status: str) -> float:
        """Get quality multiplier based on task status"""
        if not status or pd.isna(status):
            return 0.0
            
        status_lower = status.lower()
        
        for key, multiplier in self.quality_multipliers.items():
            if key in status_lower:
                return multiplier
        
        # Default fallback
        if 'approved' in status_lower:
            return 1.0
        
        return 0.0
    
    def is_paid_task(self, status: str) -> bool:
        """Check if a task is paid (not waiting for review)"""
        if not status or pd.isna(status):
            return False
        return 'waiting' not in status.lower()
    
    def parse_date_flexible(self, date_str: str) -> Optional[datetime]:
        """Parse dates in various formats"""
        if not date_str or pd.isna(date_str):
            return None
            
        date_str = str(date_str).strip()
        
        # Common formats to try
        formats = [
            "%b %d, %Y, %I:%M %p",  # Aug 24, 2025, 10:00 PM
            "%b %d, %Y, %I:%M %p",  # Aug 24, 2025, 10:00 AM
            "%b %d, %Y",            # Aug 24, 2025
            "%Y-%m-%d",             # 2025-08-24
            "%m/%d/%Y",             # 08/24/2025
            "%d/%m/%Y",             # 24/08/2025
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def parse_task_data(self, data_text: str) -> pd.DataFrame:
        """Parse pasted Excel data into a DataFrame"""
        lines = [line.strip() for line in data_text.strip().split('\n') if line.strip()]
        
        tasks = []
        for line in lines:
            # Handle both tab-separated and other delimiters
            if '\t' in line:
                parts = line.split('\t')
            else:
                # Try other common delimiters
                parts = re.split(r'[;,|]', line)
            
            if len(parts) >= 6:
                # Clean up time string
                time_str = parts[5].strip()
                time_str = re.sub(r'\s*view\s*task\s*$', '', time_str, flags=re.IGNORECASE)
                
                task = {
                    'task_id': parts[0].strip(),
                    'project': parts[1].strip(),
                    'status': parts[2].strip(),
                    'finished': parts[3].strip(),
                    'reviewed': parts[4].strip() if parts[4].strip() else None,
                    'time_str': time_str,
                    'hours': self.parse_time_string(time_str)
                }
                
                # Parse dates
                task['finished_date'] = self.parse_date_flexible(task['finished'])
                task['reviewed_date'] = self.parse_date_flexible(task['reviewed']) if task['reviewed'] else None
                
                # Calculate review delay
                if task['finished_date'] and task['reviewed_date']:
                    task['review_delay_days'] = (task['reviewed_date'] - task['finished_date']).days
                else:
                    task['review_delay_days'] = None
                
                tasks.append(task)
        
        return pd.DataFrame(tasks)
    
    def calculate_payment(self, df: pd.DataFrame, base_rate: float) -> pd.DataFrame:
        """Calculate payments for tasks"""
        df = df.copy()
        
        # Calculate multipliers and payments
        df['multiplier'] = df['status'].apply(self.get_quality_multiplier)
        df['payment'] = df['hours'] * base_rate * df['multiplier']
        df['is_paid'] = df['status'].apply(self.is_paid_task)
        
        return df
    
    def get_advanced_analytics(self, df: pd.DataFrame, base_rate: float) -> Dict:
        """Get comprehensive analytics"""
        paid_df = df[df['is_paid'] == True].copy()
        
        analytics = {
            'basic_metrics': {
                'total_tasks': len(df),
                'paid_tasks': len(paid_df),
                'unpaid_tasks': len(df) - len(paid_df),
                'total_hours': paid_df['hours'].sum(),
                'total_payment': paid_df['payment'].sum(),
                'average_hourly_rate': paid_df['payment'].sum() / paid_df['hours'].sum() if paid_df['hours'].sum() > 0 else 0,
                'base_rate': base_rate
            }
        }
        
        # Time-based analytics
        if 'finished_date' in df.columns and not df['finished_date'].isna().all():
            finished_df = df[df['finished_date'].notna()].copy()
            finished_df['week'] = finished_df['finished_date'].dt.isocalendar().week
            finished_df['month'] = finished_df['finished_date'].dt.month
            finished_df['weekday'] = finished_df['finished_date'].dt.day_name()
            
            analytics['time_patterns'] = {
                'tasks_per_week': finished_df.groupby('week').size().to_dict(),
                'tasks_per_month': finished_df.groupby('month').size().to_dict(),
                'tasks_per_weekday': finished_df.groupby('weekday').size().to_dict(),
                'hours_per_week': finished_df.groupby('week')['hours'].sum().to_dict(),
                'most_productive_day': finished_df.groupby('weekday')['hours'].sum().idxmax() if len(finished_df) > 0 else None,
                'average_tasks_per_week': len(finished_df) / max(1, finished_df['week'].nunique()),
                'average_hours_per_week': finished_df['hours'].sum() / max(1, finished_df['week'].nunique())
            }
        
        # Review delay analytics
        review_delay_df = df[df['review_delay_days'].notna()].copy()
        if len(review_delay_df) > 0:
            analytics['review_delays'] = {
                'average_delay': review_delay_df['review_delay_days'].mean(),
                'median_delay': review_delay_df['review_delay_days'].median(),
                'min_delay': review_delay_df['review_delay_days'].min(),
                'max_delay': review_delay_df['review_delay_days'].max(),
                'delays_over_30_days': len(review_delay_df[review_delay_df['review_delay_days'] > 30]),
                'delays_over_60_days': len(review_delay_df[review_delay_df['review_delay_days'] > 60])
            }
        
        # Quality analytics
        quality_stats = {}
        for status in df['status'].unique():
            status_df = df[df['status'] == status]
            quality_stats[status] = {
                'count': len(status_df),
                'hours': status_df['hours'].sum(),
                'payment': status_df['payment'].sum(),
                'avg_hours': status_df['hours'].mean(),
                'percentage': (len(status_df) / len(df)) * 100
            }
        
        analytics['quality_breakdown'] = quality_stats
        
        # Payment forecasting
        if len(paid_df) > 0:
            unpaid_df = df[df['is_paid'] == False].copy()
            # Pending payment is just hours * base rate (no quality multiplier yet)
            potential_payment = (unpaid_df['hours'] * base_rate).sum()
            # Best case scenario: if all pending tasks are approved perfectly (1.2 multiplier)
            best_case_payment = (unpaid_df['hours'] * base_rate * 1.2).sum()
            
            analytics['forecasting'] = {
                'pending_payment': potential_payment,
                'best_case_payment': best_case_payment,
                'pending_hours': unpaid_df['hours'].sum(),
                'pending_tasks': len(unpaid_df),
                'monthly_average': paid_df.groupby(paid_df['finished_date'].dt.month)['payment'].sum().mean() if 'finished_date' in paid_df.columns else 0
            }
        
        return analytics

def main():
    st.title("📊 LLM Training Payment Analytics Dashboard")
    st.markdown("*Advanced insights into your payment patterns and performance metrics*")
    
    # Initialize calculator
    calc = PaymentAnalytics()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        base_rate = st.number_input("Base Hourly Rate ($)", min_value=0.01, value=30.00, step=0.01)
        
        st.markdown("### 📊 Quality Multipliers")
        for status, multiplier in calc.quality_multipliers.items():
            if multiplier > 0:
                emoji = "🟢" if multiplier >= 1.2 else "🟡" if multiplier >= 1.0 else "🔴"
                st.markdown(f"{emoji} **{status.title()}**: {int(multiplier * 100)}%")
        
        st.markdown("### 📈 Dashboard Sections")
        show_overview = st.checkbox("📋 Overview", value=True)
        show_analytics = st.checkbox("📊 Analytics", value=True)
        show_trends = st.checkbox("📈 Trends", value=True)
        show_quality = st.checkbox("🎯 Quality Analysis", value=True)
        show_delays = st.checkbox("⏱️ Review Delays", value=True)
        show_forecasting = st.checkbox("🔮 Forecasting", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Data Input")
        task_data = st.text_area(
            "Paste Your Excel Data Here:",
            height=200,
            help="Copy all rows from Excel and paste them here. The tool will automatically parse tab-separated data.",
            placeholder="TaskID    Project    Status    Finished    Reviewed    Time"
        )
    
    with col2:
        st.markdown("""
        ### 💡 Quick Tips
        - **Copy directly** from Task History
        - **Include all columns**: TaskID, Project, Status, Finished, Reviewed, Time
        - **Date formats** are automatically detected
        - **Time formats** like "1h 30m", "45 minutes" work perfectly
        """)
    
    if task_data:
        try:
            # Parse and process data
            with st.spinner("🔄 Processing your data..."):
                df = calc.parse_task_data(task_data)
                
                if not df.empty:
                    df = calc.calculate_payment(df, base_rate)
                    analytics = calc.get_advanced_analytics(df, base_rate)
                    
                    # Store in session state
                    st.session_state.task_df = df
                    st.session_state.analytics = analytics
                    
                    st.success(f"✅ Successfully processed {len(df)} tasks!")
                    
                    # Debug: Show what data was parsed
                    try:
                        st.markdown(f"""
                        <div class="debug-info">
                            <strong>Debug: Parsed Data Preview</strong><br>
                            DataFrame shape: {df.shape}<br>
                            Columns: {list(df.columns)}<br>
                            Status values: {df['status'].unique().tolist()}<br>
                            Full dataset:
                        </div>
                        """, unsafe_allow_html=True)
                        st.dataframe(df, use_container_width=True)
                    except Exception as debug_error:
                        st.warning(f"Debug info error: {str(debug_error)}")
                        st.write("DataFrame shape:", df.shape)
                        st.write("Columns:", list(df.columns))
                        st.write("Status values:", df['status'].unique().tolist())
                        st.write("Full dataset:")
                        st.dataframe(df, use_container_width=True)
                    
                    # Overview Section
                    if show_overview:
                        st.header("📊 Payment Overview")
                        
                        metrics = analytics['basic_metrics']
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("💼 Total Tasks", metrics['total_tasks'])
                        with col2:
                            st.metric("✅ Paid Tasks", metrics['paid_tasks'])
                        with col3:
                            st.metric("⏳ Pending", metrics['unpaid_tasks'])
                        with col4:
                            st.metric("⏰ Total Hours", f"{metrics['total_hours']:.1f}")
                        with col5:
                            st.metric("💰 Total Earned", f"${metrics['total_payment']:.2f}")
                        
                        # Key insights
                        effective_rate = metrics['average_hourly_rate']
                        efficiency = (effective_rate / base_rate) * 100 if base_rate > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>${effective_rate:.2f}/hour</h3>
                                <p>Effective Rate</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{efficiency:.1f}%</h3>
                                <p>Quality Efficiency</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            paid_percentage = (metrics['paid_tasks'] / metrics['total_tasks']) * 100
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{paid_percentage:.1f}%</h3>
                                <p>Tasks Paid</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Analytics Section
                    if show_analytics and 'time_patterns' in analytics:
                        st.header("📈 Productivity Analytics")
                        
                        time_data = analytics['time_patterns']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Weekly productivity chart
                            if time_data['tasks_per_week']:
                                weeks = list(time_data['tasks_per_week'].keys())
                                task_counts = list(time_data['tasks_per_week'].values())
                                
                                fig = px.bar(
                                    x=weeks, 
                                    y=task_counts,
                                    title="📅 Tasks Completed per Week",
                                    labels={'x': 'Week Number', 'y': 'Tasks Completed'},
                                    color=task_counts,
                                    color_continuous_scale='viridis'
                                )
                                fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Daily productivity pattern
                            if time_data['tasks_per_weekday']:
                                days = list(time_data['tasks_per_weekday'].keys())
                                day_counts = list(time_data['tasks_per_weekday'].values())
                                
                                fig = px.bar(
                                    x=days, 
                                    y=day_counts,
                                    title="📊 Tasks by Day of Week",
                                    labels={'x': 'Day', 'y': 'Tasks'},
                                    color=day_counts,
                                    color_continuous_scale='plasma'
                                )
                                fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Insights
                        if time_data['most_productive_day']:
                            st.markdown(f"""
                            <div class="insight-box">
                                <strong>🎯 Productivity Insight:</strong> Your most productive day is <strong>{time_data['most_productive_day']}</strong> 
                                with an average of <strong>{time_data['average_hours_per_week']:.1f} hours/week</strong> and 
                                <strong>{time_data['average_tasks_per_week']:.1f} tasks/week</strong>.
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Quality Analysis
                    if show_quality:
                        st.header("🎯 Quality Performance Analysis")
                        
                        quality_data = analytics['quality_breakdown']
                        
                        # Quality distribution pie chart
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            statuses = list(quality_data.keys())
                            counts = [quality_data[s]['count'] for s in statuses]
                            
                            # Additional debugging
                            st.markdown(f"""
                            <div class="debug-info">
                                <strong>Raw Data Debug:</strong><br>
                                Quality data keys: {list(quality_data.keys())}<br>
                                First status: {statuses[0] if statuses else 'None'}<br>
                                First count raw: {quality_data[statuses[0]]['count'] if statuses else 'None'}<br>
                                Counts list: {counts}<br>
                                Counts length: {len(counts)}<br>
                                All counts > 0: {all(c > 0 for c in counts)}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Debug information
                            st.markdown(f"""
                            <div class="debug-info">
                                <strong>Debug Info:</strong><br>
                                Statuses found: {statuses}<br>
                                Counts: {counts}<br>
                                Total tasks: {sum(counts)}<br>
                                Counts type: {type(counts)}<br>
                                First count value: {counts[0]} (type: {type(counts[0])})
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Ensure counts are numeric and convert to list if needed
                            counts_numeric = [float(c) for c in counts]
                            
                            # Create pie chart with explicit data
                            fig = px.pie(
                                values=counts_numeric,
                                names=statuses,
                                title="📊 Task Status Distribution",
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            
                            # Add debug info to the chart
                            fig.add_annotation(
                                text=f"Total: {sum(counts_numeric)} tasks",
                                xref="paper", yref="paper",
                                x=0.5, y=1.02, showarrow=False,
                                font=dict(size=12, color="gray"),
                                xanchor="center"
                            )
                            
                            # Alternative approach - create DataFrame first
                            pie_df = pd.DataFrame({
                                'Status': statuses,
                                'Count': counts_numeric
                            })
                            
                            st.markdown(f"""
                            <div class="debug-info">
                                <strong>Pie Chart Data:</strong><br>
                                DataFrame: {pie_df.to_dict('records')}<br>
                                Values for chart: {pie_df['Count'].tolist()}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Try alternative pie chart creation method
                            try:
                                # Method 1: Direct px.pie
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as pie_error:
                                st.error(f"Pie chart error: {str(pie_error)}")
                                
                                # Method 2: Using go.Figure
                                try:
                                    fig2 = go.Figure(data=[go.Pie(
                                        labels=statuses,
                                        values=counts_numeric,
                                        hole=0.3
                                    )])
                                    fig2.update_layout(title="📊 Task Status Distribution (Alternative)")
                                    st.plotly_chart(fig2, use_container_width=True)
                                except Exception as go_error:
                                    st.error(f"Alternative pie chart also failed: {str(go_error)}")
                                    
                                    # Method 3: Simple bar chart as fallback
                                    st.subheader("📊 Task Status Distribution (Bar Chart Fallback)")
                                    fig3 = px.bar(
                                        x=statuses,
                                        y=counts_numeric,
                                        title="Task Counts by Status",
                                        labels={'x': 'Status', 'y': 'Count'}
                                    )
                                    st.plotly_chart(fig3, use_container_width=True)
                        
                        with col2:
                            # Quality vs Hours
                            hours_data = [quality_data[s]['hours'] for s in statuses]
                            
                            fig = px.bar(
                                x=statuses,
                                y=hours_data,
                                title="⏰ Hours by Quality Level",
                                labels={'x': 'Status', 'y': 'Hours'},
                                color=hours_data,
                                color_continuous_scale='RdYlGn'
                            )
                            fig.update_layout(
                                showlegend=False,
                                xaxis={'tickangle': 45}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Quality insights table
                        st.subheader("📋 Detailed Quality Breakdown")
                        quality_df = pd.DataFrame(quality_data).T
                        quality_df['percentage'] = quality_df['percentage'].round(1)
                        quality_df['avg_hours'] = quality_df['avg_hours'].round(2)
                        quality_df['payment'] = quality_df['payment'].round(2)
                        
                        st.dataframe(
                            quality_df[['count', 'hours', 'payment', 'avg_hours', 'percentage']]
                            .rename(columns={
                                'count': 'Tasks',
                                'hours': 'Total Hours',
                                'payment': 'Total Payment ($)',
                                'avg_hours': 'Avg Hours/Task',
                                'percentage': 'Percentage (%)'
                            }),
                            use_container_width=True
                        )
                    
                    # Review Delays Analysis
                    if show_delays and 'review_delays' in analytics:
                        st.header("⏱️ Review Delay Analysis")
                        
                        delay_data = analytics['review_delays']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📊 Average Delay", f"{delay_data['average_delay']:.1f} days")
                        with col2:
                            st.metric("📈 Median Delay", f"{delay_data['median_delay']:.1f} days")
                        with col3:
                            st.metric("⚠️ >30 Days", delay_data['delays_over_30_days'])
                        with col4:
                            st.metric("🚨 >60 Days", delay_data['delays_over_60_days'])
                        
                        # Review delay distribution
                        delay_df = df[df['review_delay_days'].notna()].copy()
                        if len(delay_df) > 0:
                            fig = px.histogram(
                                delay_df,
                                x='review_delay_days',
                                title="📊 Review Delay Distribution",
                                labels={'review_delay_days': 'Days to Review', 'count': 'Number of Tasks'},
                                nbins=20,
                                color_discrete_sequence=['#FF6B6B']
                            )
                            fig.add_vline(x=delay_data['average_delay'], line_dash="dash", 
                                        annotation_text=f"Avg: {delay_data['average_delay']:.1f} days")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Delay insights
                            if delay_data['average_delay'] > 30:
                                st.markdown(f"""
                                <div class="warning-box">
                                    <strong>⚠️ Long Review Times:</strong> Your average review delay is <strong>{delay_data['average_delay']:.1f} days</strong>. 
                                    This significantly impacts your cash flow. Consider following up on pending reviews.
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="success-box">
                                    <strong>✅ Good Review Times:</strong> Your average review delay is <strong>{delay_data['average_delay']:.1f} days</strong>. 
                                    This is reasonable for maintaining steady cash flow.
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Forecasting Section
                    if show_forecasting and 'forecasting' in analytics:
                        st.header("🔮 Payment Forecasting")
                        
                        forecast_data = analytics['forecasting']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("💰 Pending Payment", f"${forecast_data['pending_payment']:.2f}")
                        with col2:
                            st.metric("🚀 Best Case Payment", f"${forecast_data['best_case_payment']:.2f}")
                        with col3:
                            st.metric("⏰ Pending Hours", f"{forecast_data['pending_hours']:.1f}")
                        with col4:
                            st.metric("📋 Pending Tasks", forecast_data['pending_tasks'])
                        
                        # Monthly trends
                        if 'finished_date' in df.columns:
                            monthly_df = df[df['finished_date'].notna()].copy()
                            if len(monthly_df) > 0:
                                monthly_df['month_year'] = monthly_df['finished_date'].dt.to_period('M')
                                monthly_summary = monthly_df.groupby('month_year').agg({
                                    'payment': 'sum',
                                    'hours': 'sum',
                                    'task_id': 'count'
                                }).reset_index()
                                
                                monthly_summary['month_year_str'] = monthly_summary['month_year'].astype(str)
                                
                                fig = make_subplots(
                                    rows=2, cols=1,
                                    subplot_titles=('Monthly Payment Trend', 'Monthly Hours Trend'),
                                    vertical_spacing=0.1
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=monthly_summary['month_year_str'],
                                        y=monthly_summary['payment'],
                                        mode='lines+markers',
                                        name='Payment ($)',
                                        line=dict(color='#00CC96', width=3)
                                    ),
                                    row=1, col=1
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=monthly_summary['month_year_str'],
                                        y=monthly_summary['hours'],
                                        mode='lines+markers',
                                        name='Hours',
                                        line=dict(color='#AB63FA', width=3)
                                    ),
                                    row=2, col=1
                                )
                                
                                fig.update_layout(height=500, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Forecast insight
                                if len(monthly_summary) >= 2:
                                    recent_avg = monthly_summary['payment'].tail(2).mean()
                                    st.markdown(f"""
                                    <div class="insight-box">
                                        <strong>💡 Forecast:</strong> Based on recent trends, you're earning approximately 
                                        <strong>${recent_avg:.2f}/month</strong>. With <strong>${forecast_data['pending_payment']:.2f}</strong> 
                                        pending (base rate) and <strong>${forecast_data['best_case_payment']:.2f}</strong> 
                                        potential (perfect approval), your next payment cycle could be significant!
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Trends Section
                    if show_trends:
                        st.header("📈 Advanced Trends")
                        
                        # Task completion over time
                        if 'finished_date' in df.columns:
                            time_df = df[df['finished_date'].notna()].copy()
                            if len(time_df) > 0:
                                time_df = time_df.sort_values('finished_date')
                                time_df['cumulative_tasks'] = range(1, len(time_df) + 1)
                                time_df['cumulative_hours'] = time_df['hours'].cumsum()
                                time_df['cumulative_payment'] = time_df['payment'].cumsum()
                                
                                fig = make_subplots(
                                    rows=3, cols=1,
                                    subplot_titles=('Cumulative Tasks', 'Cumulative Hours', 'Cumulative Payment'),
                                    vertical_spacing=0.08
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=time_df['finished_date'],
                                        y=time_df['cumulative_tasks'],
                                        mode='lines',
                                        name='Tasks',
                                        line=dict(color='#636EFA', width=2)
                                    ),
                                    row=1, col=1
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=time_df['finished_date'],
                                        y=time_df['cumulative_hours'],
                                        mode='lines',
                                        name='Hours',
                                        line=dict(color='#EF553B', width=2)
                                    ),
                                    row=2, col=1
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=time_df['finished_date'],
                                        y=time_df['cumulative_payment'],
                                        mode='lines',
                                        name='Payment',
                                        line=dict(color='#00CC96', width=2)
                                    ),
                                    row=3, col=1
                                )
                                
                                fig.update_layout(height=600, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Export options
                    st.header("📥 Export Data")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Full Dataset",
                            data=csv_data,
                            file_name="payment_analysis.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        paid_csv = df[df['is_paid'] == True].to_csv(index=False)
                        st.download_button(
                            label="💰 Download Paid Tasks Only",
                            data=paid_csv,
                            file_name="paid_tasks.csv",
                            mime="text/csv"
                        )
                    
                    with col3:
                        # Analytics summary
                        analytics_summary = {
                            'summary': analytics['basic_metrics'],
                            'quality_breakdown': analytics['quality_breakdown'],
                            'review_delays': analytics.get('review_delays', {}),
                            'forecasting': analytics.get('forecasting', {})
                        }
                        analytics_json = pd.DataFrame([analytics_summary]).to_json(orient='records', indent=2)
                        st.download_button(
                            label="📈 Download Analytics Report",
                            data=analytics_json,
                            file_name="analytics_report.json",
                            mime="application/json"
                        )
                
                else:
                    st.warning("No valid task data found. Please check your input format.")
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.info("Make sure your data follows the format: TaskID | Project | Status | Finished | Reviewed | Time")
    
    else:
        st.info("👆 Paste your Excel data above to start analyzing your payment patterns!")
        
        # Sample data section
        with st.expander("📋 See Sample Data Format"):
            sample_data = """25812a00-17bd-4a14-b464-4b2c4e4fb8cd	SWE Full-trace	Approved perfectly	Jul 30, 2025, 12:00 AM	Aug 29, 2025, 5:00 AM	7h 30m	View Task
add5e603-74da-48f5-9397-ab0c12a47b26	SWE Full-trace	Approved perfectly	Aug 10, 2025, 12:00 PM	Aug 14, 2025, 6:00 AM	1h 19m	View Task
532fc3ff-4cb9-43a7-8674-94d3fe9e0e25	SWE Full-trace	Approved perfectly	Jul 20, 2025, 9:00 PM	Aug 1, 2025, 8:00 AM	1h 45m	View Task"""
            
            st.code(sample_data, language=None)
            st.markdown("Copy this sample data to test the tool!")
    
    
    # Additional Features Section
    st.header("🚀 Additional Features")
    
    if 'task_df' in st.session_state and not st.session_state.task_df.empty:
        tab1, tab2, tab3 = st.tabs(["🎯 Task Efficiency", "💡 Optimization Tips", "📊 Comparative Analysis"])
        
        with tab1:
            st.subheader("Task Efficiency Analysis")
            df = st.session_state.task_df
            
            # Efficiency metrics
            paid_df = df[df['is_paid'] == True].copy()
            
            if len(paid_df) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Hours vs Payment scatter - Fixed version
                    try:
                        fig = px.scatter(
                            paid_df,
                            x='hours',
                            y='payment',
                            color='status',
                            size='hours',  # Use hours for size instead of multiplier
                            hover_data=['task_id'],
                            title="💰 Payment vs Hours by Quality",
                            labels={'hours': 'Hours Worked', 'payment': 'Payment ($)'},
                            size_max=20
                        )
                        fig.update_traces(marker=dict(sizemin=5))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating payment scatter plot: {str(e)}")
                        
                        # Fallback: Simple bar chart
                        status_summary = paid_df.groupby('status').agg({
                            'payment': 'sum',
                            'hours': 'sum'
                        }).reset_index()
                        
                        fig = px.bar(
                            status_summary,
                            x='status',
                            y='payment',
                            title="💰 Total Payment by Status",
                            labels={'status': 'Status', 'payment': 'Total Payment ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Daily tasks timeline - NEW CHART
                    if 'finished_date' in df.columns and not df['finished_date'].isna().all():
                        daily_df = df[df['finished_date'].notna()].copy()
                        
                        # Create daily task counts
                        daily_counts = daily_df.groupby(daily_df['finished_date'].dt.date).size().reset_index()
                        daily_counts.columns = ['date', 'task_count']
                        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
                        
                        # Create complete date range from first to last task
                        date_range = pd.date_range(
                            start=daily_counts['date'].min(),
                            end=daily_counts['date'].max(),
                            freq='D'
                        )
                        
                        # Fill in missing dates with 0 tasks
                        complete_dates = pd.DataFrame({'date': date_range})
                        daily_complete = complete_dates.merge(daily_counts, on='date', how='left')
                        daily_complete['task_count'] = daily_complete['task_count'].fillna(0)
                        
                        fig = px.line(
                            daily_complete,
                            x='date',
                            y='task_count',
                            title="📅 Daily Task Completion Timeline",
                            labels={'date': 'Date', 'task_count': 'Tasks Completed'},
                            markers=True
                        )
                        
                        # Add scatter points for actual task days
                        fig.add_scatter(
                            x=daily_counts['date'],
                            y=daily_counts['task_count'],
                            mode='markers',
                            marker=dict(size=8, color='red'),
                            name='Task Days',
                            showlegend=True
                        )
                        
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Number of Tasks",
                            hovermode='x unified'
                        )
                        
                        # Add subtitle
                        fig.add_annotation(
                            text="Daily task completion from your first to most recent task - red dots show active days",
                            xref="paper", yref="paper",
                            x=0.5, y=1.12, showarrow=False,
                            font=dict(size=12, color="gray"),
                            xanchor="center"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show some stats about daily productivity
                        avg_tasks_per_day = daily_counts['task_count'].mean()
                        max_tasks_day = daily_counts.loc[daily_counts['task_count'].idxmax()]
                        total_active_days = len(daily_counts)
                        
                        st.markdown(f"""
                        **📊 Daily Productivity Stats:**
                        - **Average tasks per active day**: {avg_tasks_per_day:.1f}
                        - **Most productive day**: {max_tasks_day['date'].strftime('%Y-%m-%d')} ({int(max_tasks_day['task_count'])} tasks)
                        - **Active days**: {total_active_days} out of {len(complete_dates)} total days
                        """)
                    
                    else:
                        st.info("📅 Daily timeline requires valid finished dates in your data.")
                
                # Efficiency insights
                if len(paid_df) > 0:
                    paid_df['hourly_rate'] = paid_df['payment'] / paid_df['hours']
                    high_efficiency = paid_df[paid_df['hourly_rate'] > base_rate * 1.1]
                    low_efficiency = paid_df[paid_df['hourly_rate'] < base_rate * 0.9]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 High Efficiency Tasks", len(high_efficiency))
                        if len(high_efficiency) > 0:
                            avg_rate = high_efficiency['hourly_rate'].mean()
                            st.success(f"Average rate: ${avg_rate:.2f}/hour")
                    
                    with col2:
                        st.metric("⚠️ Low Efficiency Tasks", len(low_efficiency))
                        if len(low_efficiency) > 0:
                            avg_rate = low_efficiency['hourly_rate'].mean()
                            st.warning(f"Average rate: ${avg_rate:.2f}/hour")
                    
                    with col3:
                        overall_rate = paid_df['hourly_rate'].mean()
                        st.metric("📊 Overall Average Rate", f"${overall_rate:.2f}/hour")
                        efficiency_pct = (overall_rate / base_rate) * 100
                        st.info(f"Efficiency: {efficiency_pct:.1f}% of base rate")
            
            else:
                st.warning("No paid tasks found to analyze efficiency.")
        
        with tab2:
            st.subheader("💡 Optimization Recommendations")
            
            analytics = st.session_state.analytics
            recommendations = []
            
            # Quality-based recommendations
            quality_data = analytics['quality_breakdown']
            perfect_percentage = quality_data.get('Approved perfectly', {}).get('percentage', 0)
            
            if perfect_percentage < 50:
                recommendations.append({
                    'type': 'Quality Improvement',
                    'icon': '🎯',
                    'title': 'Increase Perfect Approval Rate',
                    'description': f'Currently {perfect_percentage:.1f}% of tasks are approved perfectly. Focus on quality to earn 20% more per task.',
                    'impact': 'High'
                })
            
            # Review delay recommendations
            if 'review_delays' in analytics:
                avg_delay = analytics['review_delays']['average_delay']
                if avg_delay > 21:
                    recommendations.append({
                        'type': 'Process Optimization',
                        'icon': '⏰',
                        'title': 'Follow Up on Reviews',
                        'description': f'Average review time is {avg_delay:.1f} days. Consider following up on pending reviews to improve cash flow.',
                        'impact': 'Medium'
                    })
            
            # Productivity recommendations
            if 'time_patterns' in analytics:
                avg_weekly_tasks = analytics['time_patterns']['average_tasks_per_week']
                if avg_weekly_tasks < 3:
                    recommendations.append({
                        'type': 'Productivity',
                        'icon': '📈',
                        'title': 'Increase Weekly Output',
                        'description': f'Currently completing {avg_weekly_tasks:.1f} tasks/week. Consider increasing to 5-7 tasks for better earnings.',
                        'impact': 'High'
                    })
            
            # Display recommendations
            for rec in recommendations:
                impact_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}[rec['impact']]
                impact_class = {'High': 'recommendation-high', 'Medium': 'recommendation-medium', 'Low': 'recommendation-low'}[rec['impact']]
                
                st.markdown(f"""
                <div class="recommendation-box {impact_class}">
                    <h4 style="margin-bottom: 0.5rem;">{rec['icon']} {rec['title']} {impact_color}</h4>
                    <p style="margin-bottom: 0.5rem;">{rec['description']}</p>
                    <small style="color: #6b7280;"><strong>Impact:</strong> {rec['impact']} | <strong>Category:</strong> {rec['type']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.subheader("📊 Comparative Analysis")
            
            df = st.session_state.task_df
            
            # Create benchmark comparisons
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏆 Your Performance vs Benchmarks")
                
                # Assumed industry benchmarks
                benchmarks = {
                    'Perfect Rate': {'your': perfect_percentage, 'benchmark': 70, 'unit': '%'},
                    'Avg Hours/Task': {'your': df['hours'].mean(), 'benchmark': 3.5, 'unit': 'hours'},
                    'Review Time': {'your': analytics.get('review_delays', {}).get('average_delay', 0), 'benchmark': 14, 'unit': 'days'},
                    'Tasks/Week': {'your': analytics.get('time_patterns', {}).get('average_tasks_per_week', 0), 'benchmark': 5, 'unit': 'tasks'}
                }
                
                comparison_data = []
                for metric, data in benchmarks.items():
                    comparison_data.append({
                        'Metric': metric,
                        'Your Performance': f"{data['your']:.1f} {data['unit']}",
                        'Industry Benchmark': f"{data['benchmark']:.1f} {data['unit']}",
                        'Performance': 'Above' if data['your'] >= data['benchmark'] else 'Below'
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Monthly Performance Trends")
                
                if 'finished_date' in df.columns:
                    monthly_df = df[df['finished_date'].notna()].copy()
                    if len(monthly_df) > 0:
                        monthly_df['month'] = monthly_df['finished_date'].dt.to_period('M')
                        monthly_stats = monthly_df.groupby('month').agg({
                            'payment': 'sum',
                            'hours': 'sum',
                            'task_id': 'count'
                        }).reset_index()
                        
                        # Calculate month-over-month growth
                        monthly_stats['payment_growth'] = monthly_stats['payment'].pct_change() * 100
                        monthly_stats['hours_growth'] = monthly_stats['hours'].pct_change() * 100
                        
                        if len(monthly_stats) > 1:
                            latest_payment_growth = monthly_stats['payment_growth'].iloc[-1]
                            latest_hours_growth = monthly_stats['hours_growth'].iloc[-1]
                            
                            st.metric(
                                "Payment Growth (MoM)",
                                f"{latest_payment_growth:.1f}%",
                                delta=f"{latest_payment_growth:.1f}%"
                            )
                            st.metric(
                                "Hours Growth (MoM)", 
                                f"{latest_hours_growth:.1f}%",
                                delta=f"{latest_hours_growth:.1f}%"
                            )
    
    # Footer with additional info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>📊 <strong>LLM Training Payment Analytics</strong></p>
        <p>💡 <em>For best results, ensure your data includes all columns: TaskID, Project, Status, Finished, Reviewed, Time</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()