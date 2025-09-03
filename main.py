import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import List, Dict, Tuple, Optional
import io

# Set page configuration
st.set_page_config(
    page_title="LLM Labelers Payment Calculator",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PaymentCalculator:
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
    
    def get_payment_summary(self, df: pd.DataFrame, base_rate: float) -> Dict:
        """Get payment summary statistics"""
        paid_tasks = df[df['is_paid'] == True]
        
        summary = {
            'total_tasks': len(df),
            'paid_tasks': len(paid_tasks),
            'unpaid_tasks': len(df) - len(paid_tasks),
            'total_hours': paid_tasks['hours'].sum(),
            'total_payment': paid_tasks['payment'].sum(),
            'base_rate': base_rate
        }
        
        # Quality breakdown
        quality_breakdown = {}
        for status_key in ['approved perfectly', 'approved with minor edits', 'approved with major fixes', 'waiting for review']:
            status_tasks = df[df['status'].str.lower().str.contains(status_key, na=False)]
            if len(status_tasks) > 0:
                quality_breakdown[status_key] = {
                    'count': len(status_tasks),
                    'hours': status_tasks['hours'].sum(),
                    'payment': status_tasks['payment'].sum(),
                    'multiplier': self.quality_multipliers[status_key]
                }
        
        summary['quality_breakdown'] = quality_breakdown
        return summary
    
    def find_matching_tasks(self, df: pd.DataFrame, invoice_date: datetime, 
                          invoice_amount: float, base_rate: float, 
                          matching_mode: str = 'review_based') -> Tuple[pd.DataFrame, Dict]:
        """Find tasks that match an invoice"""
        
        # Only consider paid tasks
        candidate_df = df[df['is_paid'] == True].copy()
        
        if matching_mode == 'review_based':
            # Look for tasks reviewed within 2 weeks before invoice date
            lookback_date = invoice_date - timedelta(days=14)
            
            def is_in_range(row):
                if not row['reviewed']:
                    return False
                try:
                    review_date = pd.to_datetime(row['reviewed'])
                    return lookback_date <= review_date <= invoice_date
                except:
                    return False
            
            candidate_df = candidate_df[candidate_df.apply(is_in_range, axis=1)]
            date_range_text = f"Tasks Reviewed: {lookback_date.strftime('%Y-%m-%d')} to {invoice_date.strftime('%Y-%m-%d')}"
            
        elif matching_mode == 'payment_cycle':
            # 7 days after review week
            work_week_end = invoice_date - timedelta(days=7)
            # Find the Monday of that week
            days_since_monday = work_week_end.weekday()
            monday = work_week_end - timedelta(days=days_since_monday)
            sunday = monday + timedelta(days=6)
            
            def is_in_work_week(row):
                if not row['reviewed']:
                    return False
                try:
                    review_date = pd.to_datetime(row['reviewed'])
                    return monday <= review_date <= sunday
                except:
                    return False
            
            candidate_df = candidate_df[candidate_df.apply(is_in_work_week, axis=1)]
            date_range_text = f"Review Week: {monday.strftime('%Y-%m-%d')} to {sunday.strftime('%Y-%m-%d')}"
            
        else:  # custom_lookback
            lookback_days = 30  # Default
            lookback_date = invoice_date - timedelta(days=lookback_days)
            
            def is_in_custom_range(row):
                date_str = row['reviewed'] if row['reviewed'] else row['finished']
                if not date_str:
                    return False
                try:
                    task_date = pd.to_datetime(date_str)
                    return lookback_date <= task_date <= invoice_date
                except:
                    return False
            
            candidate_df = candidate_df[candidate_df.apply(is_in_custom_range, axis=1)]
            date_range_text = f"Custom Range: {lookback_date.strftime('%Y-%m-%d')} to {invoice_date.strftime('%Y-%m-%d')}"
        
        # Calculate match results
        calculated_total = candidate_df['payment'].sum()
        difference = abs(calculated_total - invoice_amount)
        percent_diff = (difference / invoice_amount * 100) if invoice_amount > 0 else 100
        
        match_quality = "Perfect Match" if percent_diff <= 1 else "Close Match" if percent_diff <= 5 else "Significant Difference"
        
        results = {
            'date_range': date_range_text,
            'calculated_total': calculated_total,
            'invoice_amount': invoice_amount,
            'difference': difference,
            'percent_diff': percent_diff,
            'match_quality': match_quality,
            'task_count': len(candidate_df)
        }
        
        return candidate_df, results

def main():
    st.title("🧮 LLM Training Payment Calculator & Invoice Matcher")
    
    # Initialize calculator
    calc = PaymentCalculator()
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    base_rate = st.sidebar.number_input("Base Hourly Rate ($)", min_value=0.01, value=30.00, step=0.01)
    
    # Quality multiplier info
    st.sidebar.markdown("### 📊 Quality Multipliers")
    for status, multiplier in calc.quality_multipliers.items():
        if multiplier > 0:
            emoji = "✅" if multiplier >= 1.2 else "⚠️" if multiplier >= 1.0 else "❌"
            st.sidebar.markdown(f"{emoji} {status.title()}: {int(multiplier * 100)}%")
    
    # Payment reality info
    st.info("""
    **💡 Payment Reality**: You get paid only AFTER tasks are reviewed and approved. 
    Tasks can take weeks/months between finishing and review. The tool accounts for these delays.
    """)
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📊 Task Calculator", "🧾 Invoice Matcher"])
    
    with tab1:
        st.header("📊 Task Calculator")
        
        # Input area for task data
        st.markdown("### Paste Your Excel Data")
        task_data = st.text_area(
            "Copy and paste your task data from Excel (one task per line):",
            height=200,
            help="Copy all rows from Excel and paste them here. The tool will automatically parse tab-separated data."
        )
        
        if task_data:
            try:
                # Parse the data
                df = calc.parse_task_data(task_data)
                
                if not df.empty:
                    # Calculate payments
                    df = calc.calculate_payment(df, base_rate)
                    
                    # Display summary
                    summary = calc.get_payment_summary(df, base_rate)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Tasks", summary['total_tasks'])
                    with col2:
                        st.metric("Paid Tasks", summary['paid_tasks'])
                    with col3:
                        st.metric("Total Hours", f"{summary['total_hours']:.2f}")
                    with col4:
                        st.metric("Total Payment", f"${summary['total_payment']:.2f}")
                    
                    # Quality breakdown
                    if summary['quality_breakdown']:
                        st.markdown("### 📈 Breakdown by Quality")
                        breakdown_data = []
                        for status, data in summary['quality_breakdown'].items():
                            breakdown_data.append({
                                'Status': status.title(),
                                'Count': data['count'],
                                'Hours': f"{data['hours']:.2f}",
                                'Payment': f"${data['payment']:.2f}",
                                'Multiplier': f"{int(data['multiplier'] * 100)}%"
                            })
                        
                        breakdown_df = pd.DataFrame(breakdown_data)
                        st.dataframe(breakdown_df, use_container_width=True)
                    
                    # Detailed task list
                    st.markdown("### 📋 Task Details")
                    
                    # Prepare display dataframe
                    display_df = df.copy()
                    display_df['Task ID (Short)'] = display_df['task_id'].str[:8] + '...'
                    display_df['Hours'] = display_df['hours'].round(2)
                    display_df['Payment'] = display_df['payment'].round(2)
                    display_df['Paid Status'] = display_df['is_paid'].map({True: '✅ Paid', False: '⏳ Unpaid'})
                    
                    # Select columns for display
                    display_cols = ['Task ID (Short)', 'status', 'time_str', 'Hours', 'Payment', 'Paid Status', 'finished', 'reviewed']
                    st.dataframe(
                        display_df[display_cols].rename(columns={
                            'status': 'Status',
                            'time_str': 'Time',
                            'finished': 'Finished',
                            'reviewed': 'Reviewed'
                        }), 
                        use_container_width=True
                    )
                    
                    # Store data in session state for invoice matching
                    st.session_state.task_df = df
                    
                else:
                    st.warning("No valid task data found. Please check your input format.")
                    
            except Exception as e:
                st.error(f"Error parsing task data: {str(e)}")
                st.info("Make sure your data is in the format: TaskID | Project | Status | Finished | Reviewed | Time")
    
    with tab2:
        st.header("🧾 Invoice Matcher")
        
        if 'task_df' not in st.session_state or st.session_state.task_df.empty:
            st.warning("⚠️ Please add task data in the Task Calculator tab first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                invoice_date = st.date_input("Invoice Date", value=datetime(2025, 8, 17))
                invoice_amount = st.number_input("Invoice Amount ($)", min_value=0.01, value=302.50, step=0.01)
            
            with col2:
                matching_mode = st.selectbox(
                    "Matching Strategy",
                    ['review_based', 'payment_cycle', 'custom_lookback'],
                    format_func=lambda x: {
                        'review_based': 'Review Date Based (Recommended)',
                        'payment_cycle': 'Payment Cycle (7 days after review)',
                        'custom_lookback': 'Custom Lookback Days'
                    }[x]
                )
            
            if st.button("🔍 Find Matching Tasks", type="primary"):
                try:
                    invoice_datetime = datetime.combine(invoice_date, datetime.min.time())
                    matching_tasks, results = calc.find_matching_tasks(
                        st.session_state.task_df, 
                        invoice_datetime, 
                        invoice_amount, 
                        base_rate, 
                        matching_mode
                    )
                    
                    # Display results
                    st.markdown("### 🔍 Match Results")
                    
                    # Match quality indicator
                    if results['match_quality'] == "Perfect Match":
                        st.success(f"✅ {results['match_quality']}")
                    elif results['match_quality'] == "Close Match":
                        st.warning(f"⚠️ {results['match_quality']}")
                    else:
                        st.error(f"❌ {results['match_quality']}")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Invoice Amount", f"${results['invoice_amount']:.2f}")
                    with col2:
                        st.metric("Calculated Amount", f"${results['calculated_total']:.2f}")
                    with col3:
                        st.metric("Difference", f"${results['difference']:.2f}")
                    with col4:
                        st.metric("Percent Diff", f"{results['percent_diff']:.1f}%")
                    
                    st.info(f"**Search Strategy**: {results['date_range']}")
                    
                    # Matching tasks table
                    if not matching_tasks.empty:
                        st.markdown(f"### 📋 Matching Tasks ({len(matching_tasks)})")
                        
                        # Prepare display
                        display_tasks = matching_tasks.copy()
                        display_tasks['Task ID'] = display_tasks['task_id'].str[:12] + '...'
                        display_tasks['Hours'] = display_tasks['hours'].round(2)
                        display_tasks['Payment'] = display_tasks['payment'].apply(lambda x: f"${x:.2f}")
                        display_tasks['Finished Date'] = pd.to_datetime(display_tasks['finished']).dt.strftime('%Y-%m-%d')
                        display_tasks['Reviewed Date'] = display_tasks['reviewed'].apply(
                            lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if x else 'Not reviewed'
                        )
                        
                        display_cols = ['Task ID', 'status', 'time_str', 'Hours', 'Payment', 'Finished Date', 'Reviewed Date']
                        st.dataframe(
                            display_tasks[display_cols].rename(columns={
                                'status': 'Status',
                                'time_str': 'Time Worked'
                            }),
                            use_container_width=True
                        )
                        
                        # Export option
                        csv_buffer = io.StringIO()
                        matching_tasks.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Matching Tasks CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"matching_tasks_{invoice_date}.csv",
                            mime="text/csv"
                        )
                        
                    else:
                        st.warning("❌ No matching tasks found in this date range.")
                        st.info("💡 Try switching to 'Custom Lookback Days' with a longer range, or check if review dates are missing.")
                
                except Exception as e:
                    st.error(f"Error matching invoice: {str(e)}")

if __name__ == "__main__":
    main()