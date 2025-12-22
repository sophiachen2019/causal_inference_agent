import streamlit as st

import pandas as pd
import numpy as np
import numpy as np
# import dowhy # Unused directly
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from econml.dml import LinearDML, CausalForestDML

import matplotlib.pyplot as plt
from dowhy import CausalModel

from scipy import stats
import statsmodels.api as sm

def get_index(columns, default_name, default_idx):
    if default_name in columns:
        return list(columns).index(default_name)
    return default_idx if default_idx < len(columns) else 0


# Import custom utils with explicit reload to ensure updates are picked up
import causal_utils
import importlib
importlib.reload(causal_utils)
from causal_utils import generate_script

# --- 1. Data Simulation ---
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'bucketing_ops' not in st.session_state:
    st.session_state.bucketing_ops = []
if 'filtering_ops' not in st.session_state:
    st.session_state.filtering_ops = []

@st.cache_data
def simulate_data(n_samples=1000):
    np.random.seed(42)
    
    # Confounders
    # Customer Segment: 0 = SMB, 1 = Enterprise
    customer_segment = np.random.binomial(1, 0.3, n_samples)
    
    # Historical Usage: Continuous variable
    historical_usage = np.random.normal(50, 15, n_samples) + (customer_segment * 20)
    
    # Instrument: Marketing Nudge (Randomly assigned, affects adoption but not value directly)
    marketing_nudge = np.random.binomial(1, 0.5, n_samples)
    
    # Time Period: Quarter (0 = Pre, 1 = Post) - for DiD
    quarter = np.random.binomial(1, 0.5, n_samples)

    # Treatment: Feature Adoption (Binary)
    # Probability of adoption depends on segment, usage, AND marketing nudge
    prob_adoption = 1 / (1 + np.exp(-( -2 + 0.5 * customer_segment + 0.05 * historical_usage + 1.5 * marketing_nudge)))
    feature_adoption = np.random.binomial(1, prob_adoption, n_samples)
    
    # Outcome: Account Value
    # True causal effect of feature adoption is $500
    # Also depends on segment, usage, and time (trend)
    account_value = (
        200 
        + 500 * feature_adoption 
        + 1000 * customer_segment 
        + 10 * historical_usage 
        + 50 * quarter # Time trend
        + np.random.normal(0, 50, n_samples)
    )

    # Outcome: Conversion (Binary)
    # Base prob depends on segment. Treatment increases prob by ~10%
    prob_conversion = 1 / (1 + np.exp(-( -1 + 0.5 * customer_segment + 0.5 * feature_adoption)))
    conversion = np.random.binomial(1, prob_conversion, n_samples)
    
    df = pd.DataFrame({
        'Customer_Segment': customer_segment,
        'Historical_Usage': historical_usage,
        'Marketing_Nudge': marketing_nudge,
        'Quarter': quarter,
        'Feature_Adoption': feature_adoption,
        'Account_Value': account_value,
        'Conversion': conversion
    })
    
    # Enforce Data Types
    df['Customer_Segment'] = df['Customer_Segment'].astype(int)
    df['Historical_Usage'] = df['Historical_Usage'].astype(float)
    df['Marketing_Nudge'] = df['Marketing_Nudge'].astype(int)
    df['Quarter'] = df['Quarter'].astype(int)
    df['Feature_Adoption'] = df['Feature_Adoption'].astype(int)
    df['Account_Value'] = df['Account_Value'].astype(float)
    df['Conversion'] = df['Conversion'].astype(int)
    
    return df

def convert_bool_to_int(df):
    # 1. Actual boolean types
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # 2. String "TRUE"/"FALSE" (case insensitive)
    # Check object columns
    obj_cols = df.select_dtypes(include=['object']).columns
    
    # Mapping dictionary
    mapping = {
        'TRUE': 1, 'FALSE': 0,
        'T': 1, 'F': 0,
        '1': 1, '0': 0,
        '1.0': 1, '0.0': 0
    }
    
    for col in obj_cols:
        # Optimization: Check first value or sample to decide if we should attempt conversion
        # to avoid expensive unique() on large columns.
        # Or just try to map and check if we introduced NaNs where there weren't any.
        
        # Fast check: are there any values that look like booleans?
        # Let's stick to the safe unique check but optimize it slightly?
        # Actually, for huge data, unique() is slow.
        # Let's try to convert and see if it works.
        
        try:
            # Attempt to map everything to upper case
            series_upper = df[col].astype(str).str.upper()
            
            # Check if all unique values are in our mapping keys (plus NaNs)
            # We can check a sample first for speed
            sample = series_upper.dropna().head(100)
            if not set(sample.unique()).issubset(mapping.keys()):
                continue # Skip if sample has non-boolean values
                
            # If sample passed, check full unique (still safer than blind conversion)
            # Or trust the sample? Let's check full unique but only if sample passed.
            unique_vals = set(series_upper.dropna().unique())
            if unique_vals.issubset(mapping.keys()):
                df[col] = series_upper.map(mapping).fillna(df[col])
                df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass
            
    return df

# --- Streamlit UI ---
st.set_page_config(page_title="Causal Inference Agent", layout="wide")
st.title("🤖 Causal Inference Agent")
st.markdown("**Builder:** [Sophia Chen](https://www.linkedin.com/in/sophia-chen-34794893/) | **Email:** sophiachen2012@gmail.com | **Medium:** [medium.com/@sophiachen2012](https://medium.com/@sophiachen2012)")

def get_app_metadata():
    """Parses metadata from requirements.txt"""
    history = []
    try:
        with open("requirements.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# Release:"):
                    # Format: # Release: <Version> | <Date> | <Note>
                    parts = line.split("|")
                    if len(parts) >= 3:
                        version = parts[0].split(":", 1)[1].strip()
                        date = parts[1].strip()
                        note = parts[2].strip()
                        history.append({
                            "Version": version,
                            "Release Date": date,
                            "Release Note": note
                        })
    except FileNotFoundError:
        pass
    return history

history = get_app_metadata()
latest_version = history[0] if history else {"Version": "Unknown", "Release Date": "Unknown", "Release Note": "Unknown"}

# Load Data
# --- Tabs Setup ---
tab_guide, tab_eda, tab_causal = st.tabs(["User Guide", "Exploratory Analysis", "Causal Analysis"])

# ==========================================
# TAB 2: Exploratory Analysis
# ==========================================
with tab_eda:
    st.header("Exploratory Data Analysis")
    
    # --- Data Source ---
    st.subheader("1. Data Source")
    data_source = st.radio("Data Source", ["Simulated Data", "Upload CSV"], horizontal=True)
    
    # Initialize Session State
    if 'df' not in st.session_state:
        st.session_state.df = simulate_data()
    
    if 'bucketing_ops' not in st.session_state:
        st.session_state.bucketing_ops = []
    
    if 'filtering_ops' not in st.session_state:
        st.session_state.filtering_ops = []
    


    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            # Check if it's a new file
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if 'uploaded_file_id' not in st.session_state or st.session_state.uploaded_file_id != file_id:
                # New file loaded
                # New file loaded
                df = pd.read_csv(uploaded_file)
                df = convert_bool_to_int(df) # Auto-convert booleans
                st.session_state.raw_df = df.copy() # Store raw data
                st.session_state.df = df.copy() # Initialize working df
                st.session_state.uploaded_file_id = file_id
                # Reset ops
                st.session_state.bucketing_ops = []
                st.session_state.filtering_ops = []
                st.session_state.filtering_ops = []
                st.rerun()
            else:
                # Same file, keep existing session state df (which might have edits)
                df = st.session_state.df
        else:
            st.info("Awaiting CSV upload. Using simulated data for preview.")
            # Fallback to simulated if no file, but don't overwrite if we had one before?
            # Actually, if no file is uploaded, we should probably show simulated or empty.
            if 'df' not in st.session_state:
                 st.session_state.df = simulate_data()
            df = st.session_state.df
    else:
        # Simulated Data
        # If switching to simulated, we might want to reset if we were previously on Upload
        # For now, simplistic approach: if we don't have a df, simulate it.
        if 'df' not in st.session_state:
             st.session_state.df = simulate_data()
             st.session_state.raw_df = st.session_state.df.copy()
        
        # If the user explicitly wants to reset simulated data, we can add a button
        if st.button("Reset Simulated Data"):
            st.session_state.df = simulate_data()
            st.session_state.raw_df = st.session_state.df.copy()
            st.session_state.bucketing_ops = []
            st.session_state.filtering_ops = []
            st.session_state.filtering_ops = []
            st.rerun()
            
        df = st.session_state.df

    # --- Data Preprocessing ---
    st.subheader("2. Data Preprocessing")
    with st.expander("Preprocessing Options", expanded=False):
        st.markdown("### Transformations")
        
        # 1. Missing Value Imputation
        st.markdown("#### Missing Value Imputation")
        impute_enable = st.checkbox("Enable Imputation", value=False)
        
        if impute_enable:
            col1, col2 = st.columns(2)
            with col1:
                num_impute_method = st.selectbox(
                    "Numeric Imputation Method",
                    ["Mean", "Median", "Zero", "Custom Value"]
                )
                if num_impute_method == "Custom Value":
                    num_custom_val = st.number_input("Custom Value (Numeric)", value=0.0)
            
            with col2:
                cat_impute_method = st.selectbox(
                    "Categorical Imputation Method",
                    ["Mode", "Missing Indicator", "Custom Value"]
                )
                if cat_impute_method == "Custom Value":
                    cat_custom_val = st.text_input("Custom Value (Categorical)", value="Missing")

            # Apply Imputation
            # Numeric
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                if num_impute_method == "Mean":
                    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
                elif num_impute_method == "Median":
                    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
                elif num_impute_method == "Zero":
                    df[num_cols] = df[num_cols].fillna(0)
                elif num_impute_method == "Custom Value":
                    df[num_cols] = df[num_cols].fillna(num_custom_val)
            
            # Categorical
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            if len(cat_cols) > 0:
                if cat_impute_method == "Mode":
                    for col in cat_cols:
                        if not df[col].mode().empty:
                            df[col] = df[col].fillna(df[col].mode()[0])
                elif cat_impute_method == "Missing Indicator":
                    df[cat_cols] = df[cat_cols].fillna("Missing")
                elif cat_impute_method == "Custom Value":
                    df[cat_cols] = df[cat_cols].fillna(cat_custom_val)
            
            st.info("Missing values imputed.")

        # 2. Winsorization
        st.markdown("#### Winsorization (Outlier Handling)")
        winsorize_enable = st.checkbox("Enable Winsorization", value=False)
        
        if winsorize_enable:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            winsorize_cols = st.multiselect("Select columns to winsorize", numeric_cols, default=[])
            
            if winsorize_cols:
                percentile = st.slider("Percentile Threshold", min_value=0.01, max_value=0.25, value=0.05, step=0.01, help="Clips values at the p-th and (1-p)-th percentiles.")
                
                for col in winsorize_cols:
                    lower = df[col].quantile(percentile)
                    upper = df[col].quantile(1 - percentile)
                    df[col] = df[col].clip(lower=lower, upper=upper)
                
                st.info(f"Winsorization applied to {', '.join(winsorize_cols)} at {percentile*100:.0f}% threshold.")

        # 3. Log Transformation
        st.markdown("#### Log Transformation")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        log_transform_cols = st.multiselect("Apply Log Transformation (np.log1p)", numeric_cols, help="Applies log(x+1) to selected columns.")
        
        if log_transform_cols:
            for col in log_transform_cols:
                # Ensure no negative values for log
                if (df[col] < 0).any():
                    st.warning(f"Column '{col}' contains negative values. Log transformation skipped for this column.")
                else:
                    df[col] = np.log1p(df[col])
            st.info(f"Log transformation applied to: {', '.join(log_transform_cols)}")

        # 4. Standardization
        st.markdown("#### Standardization")
        standardize_cols = st.multiselect("Standardize Variables (StandardScaler)", numeric_cols, help="Scales variables to have mean=0 and std=1.")
        
        if standardize_cols:
            scaler = StandardScaler()
            df[standardize_cols] = scaler.fit_transform(df[standardize_cols])
            st.info(f"Standardization applied to: {', '.join(standardize_cols)}")

        # 5. Variable Bucketing (Binning)
        st.markdown("#### Variable Bucketing (Binning)")
        with st.expander("Create Bucketed Variables", expanded=False):
            bucket_col = st.selectbox("Select Column to Bucket", numeric_cols, key="bucket_col")
            n_bins = st.number_input("Number of Bins", min_value=2, max_value=20, value=4, step=1, key="n_bins")
            bin_method = st.radio("Binning Method", ["Equal Width (cut)", "Equal Frequency (qcut)"], key="bin_method")
            new_col_name = st.text_input("New Column Name", value=f"{bucket_col}_binned" if bucket_col else "binned_col", key="new_col_name")
            
            if st.button("Create Buckets"):
                if bucket_col and new_col_name:
                    try:
                        if bin_method == "Equal Width (cut)":
                            st.session_state.df[new_col_name] = pd.cut(st.session_state.df[bucket_col], bins=n_bins, labels=False)
                            method_code = "cut"
                        else:
                            st.session_state.df[new_col_name] = pd.qcut(st.session_state.df[bucket_col], q=n_bins, labels=False, duplicates='drop')
                            method_code = "qcut"
                        
                        # Store operation for export script
                        op = {
                            "col": bucket_col,
                            "n_bins": n_bins,
                            "method": method_code,
                            "new_col": new_col_name
                        }
                        st.session_state.bucketing_ops.append(op)
                        
                        st.success(f"Created '{new_col_name}' with {n_bins} bins.")
                        st.write(st.session_state.df[new_col_name].value_counts().sort_index())
                        st.rerun()
                    except Exception as e:
                        st.error(f"Bucketing failed: {e}")
                else:
                    st.error("Please select a column and provide a name.")

        # 6. Data Filtering
        st.markdown("#### Data Filtering")
        with st.expander("Filter Data", expanded=False):
            filter_col = st.selectbox("Select Column to Filter", df.columns, key="filter_col")
            filter_op = st.selectbox("Condition", ["==", "!=", ">", "<", ">=", "<=", "contains"], key="filter_op")
            
            # Dynamic input based on column type
            if pd.api.types.is_numeric_dtype(df[filter_col]):
                filter_val = st.number_input("Value", value=0.0, key="filter_val_num")
            else:
                filter_val = st.text_input("Value", key="filter_val_text")
                
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if st.button("Apply Filter"):
                    try:
                        initial_len = len(st.session_state.df)
                        
                        if filter_op == "==":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col] == filter_val]
                        elif filter_op == "!=":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col] != filter_val]
                        elif filter_op == ">":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col] > filter_val]
                        elif filter_op == "<":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col] < filter_val]
                        elif filter_op == ">=":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col] >= filter_val]
                        elif filter_op == "<=":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col] <= filter_val]
                        elif filter_op == "contains":
                            st.session_state.df = st.session_state.df[st.session_state.df[filter_col].astype(str).str.contains(str(filter_val), na=False)]
                        
                        final_len = len(st.session_state.df)
                        
                        # Store operation
                        op = {
                            "col": filter_col,
                            "op": filter_op,
                            "val": filter_val
                        }
                        st.session_state.filtering_ops.append(op)
                        
                        st.success(f"Filter applied. Rows reduced from {initial_len} to {final_len}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Filtering failed: {e}")
            
            with col_f2:
                if st.button("Reset Data"):
                    # Reload from raw_df
                    if 'raw_df' in st.session_state:
                         st.session_state.df = st.session_state.raw_df.copy()
                    else:
                         st.session_state.df = simulate_data()
                         st.session_state.raw_df = st.session_state.df.copy()
                    
                    st.session_state.bucketing_ops = []
                    st.session_state.filtering_ops = []
                    st.rerun()

    # --- Data Preview ---
    st.subheader("3. Data Preview")
    st.dataframe(df.head())
    
    # --- Data Summary ---
    st.subheader("4. Data Summary")
    with st.expander("Show Summary Statistics", expanded=False):
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe())
        
        st.markdown("**Missing Values**")
        missing_info = pd.DataFrame({
            'Missing Count': df.isnull().sum(),
            'Missing Percentage': (df.isnull().sum() / len(df)) * 100
        })
        st.dataframe(missing_info.style.format({'Missing Percentage': '{:.2f}%'}))

    with st.expander("Show Correlation Matrix", expanded=False):
        st.markdown("**Correlation Matrix**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            corr_cols = st.multiselect("Select Variables", numeric_cols, default=numeric_cols)
            
            if corr_cols:
                corr_matrix = df[corr_cols].corr()
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
            else:
                st.info("Please select at least one variable.")
        else:
            st.info("Not enough numeric variables to compute correlation.")

    # --- Chart Builder ---
    st.subheader("5. Visualization (Chart Builder)")
    
    chart_type = st.selectbox("Chart Type", ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Pie Chart"])
    
    col_x, col_y, col_color = st.columns(3)
    
    with col_x:
        x_var = st.selectbox("X Variable", df.columns)
    
    with col_y:
        # Y variable is optional for some charts
        if chart_type in ["Histogram", "Pie Chart"]:
            y_var = None
        else:
            y_var = st.selectbox("Y Variable", df.columns, index=1 if len(df.columns) > 1 else 0)
            
    with col_color:
        color_var = st.selectbox("Color/Group (Optional)", [None] + list(df.columns))

    # Aggregation Options
    enable_aggregation = st.checkbox("Aggregate Data")
    if enable_aggregation:
        agg_method = st.selectbox("Aggregation Method", ["Mean", "Sum", "Count", "Median", "Min", "Max"])
        
        if y_var:
            try:
                if color_var:
                    df_plot = df.groupby([x_var, color_var])[y_var].agg(agg_method.lower()).reset_index()
                else:
                    df_plot = df.groupby(x_var)[y_var].agg(agg_method.lower()).reset_index()
                
                st.info(f"Plotting {agg_method} of {y_var} by {x_var}")
            except Exception as e:
                st.error(f"Aggregation failed: {e}")
                df_plot = df
        else:
             # For Histogram/Pie where Y might not be needed or is count
             st.warning("Aggregation is mostly relevant when a Y variable is selected (e.g., Bar/Line charts).")
             df_plot = df
    else:
        df_plot = df

    if chart_type == "Scatter Plot":
        st.scatter_chart(df_plot, x=x_var, y=y_var, color=color_var)
    elif chart_type == "Line Chart":
        st.line_chart(df_plot, x=x_var, y=y_var, color=color_var)
    elif chart_type == "Bar Chart":
        st.bar_chart(df_plot, x=x_var, y=y_var, color=color_var)
    elif chart_type == "Histogram":
        fig, ax = plt.subplots()
        if color_var:
            for label, group in df_plot.groupby(color_var):
                ax.hist(group[x_var], alpha=0.5, label=str(label), bins=20)
            ax.legend()
        else:
            ax.hist(df_plot[x_var], bins=20)
        ax.set_title(f"Histogram of {x_var}")
        st.pyplot(fig)
    elif chart_type == "Box Plot":
        fig, ax = plt.subplots()
        if color_var:
            # Boxplot with grouping
            data = []
            labels = []
            for label, group in df_plot.groupby(color_var):
                data.append(group[x_var] if y_var is None else group[y_var])
                labels.append(label)
            ax.boxplot(data, labels=labels)
        else:
            ax.boxplot(df_plot[x_var] if y_var is None else df_plot[y_var])
        st.pyplot(fig)
    elif chart_type == "Pie Chart":
        fig, ax = plt.subplots()
        if color_var:
             counts = df_plot[color_var].value_counts()
             ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
        else:
             counts = df_plot[x_var].value_counts()
             ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
        st.pyplot(fig)


# ==========================================
# TAB 3: Causal Analysis
# ==========================================
with tab_causal:
    st.header("Causal Analysis Configuration")
    
    # Ensure columns exist in df (handle case where upload might have different columns)


    estimation_method = st.selectbox(
        "Estimation Method",
        [
            "OLS/Logit",
            "Double Machine Learning (LinearDML)", 
            "Propensity Score Matching",
            "Inverse Propensity Weighting (IPTW)",
            "Meta-Learner: S-Learner",
            "Meta-Learner: T-Learner",
            "Causal Forest (DML)",
            "Difference-in-Differences (DiD)"
        ]
    )

    treatment = st.selectbox("Treatment (Action)", df.columns, index=get_index(df.columns, 'Feature_Adoption', 2))
    
    # -----------------------------------------------------------
    # Handle Categorical Treatment
    # -----------------------------------------------------------
    if df[treatment].dtype == 'object' or df[treatment].dtype.name == 'category':
        st.info(f"Detected categorical treatment: {treatment}. Encoding as binary.")
        unique_vals = df[treatment].unique()
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            control_val = st.selectbox("Control Value (0)", unique_vals, index=0)
        with col_t2:
            treat_val = st.selectbox("Treatment Value (1)", unique_vals, index=1 if len(unique_vals) > 1 else 0)
            
        if control_val == treat_val:
            st.error("Control and Treatment values must be different.")
            st.stop()
            
        # Create a temporary encoded column for analysis
        df['Treatment_Encoded'] = df[treatment].apply(lambda x: 1 if x == treat_val else 0)
        # Update treatment variable to point to new column
        treatment = 'Treatment_Encoded'
    # ----------------------------------------------------------- 
    outcome = st.selectbox("Outcome (Result)", df.columns, index=get_index(df.columns, 'Account_Value', 3))
    
    # Check for Binary Outcome
    is_binary_outcome = False
    if df[outcome].nunique() == 2:
        is_binary_outcome = True
        st.info(f"Detected binary outcome: `{outcome}`. Using Classification models for estimation.")

    default_confounders = [c for c in ['Customer_Segment', 'Historical_Usage'] if c in df.columns]
    confounders = st.multiselect("Confounders (Common Causes)", df.columns, default=default_confounders)
    
    # Optional Inputs for Advanced Methods
    time_period = st.selectbox("Time Period (for DiD)", [None] + list(df.columns), index=0)
    
    st.markdown("#### Bootstrapping Settings")
    enable_bootstrap = st.checkbox("Enable Bootstrapping (Calculate Standard Errors)", value=True)
    if enable_bootstrap:
        n_iterations = st.number_input("Bootstrap Iterations", min_value=10, max_value=500, value=50, step=10, help="Number of resampling iterations for SE estimation.")
    else:
        n_iterations = 0

    def on_run_click():
        st.session_state.analysis_run = True

    st.button("Run Causal Analysis", type="primary", on_click=on_run_click)



# ==========================================
# ==========================================
# TAB 1: User Guide
# ==========================================
with tab_guide:
    st.header("User Guide")
    
    st.markdown("""
    ## Welcome to the Causal Agent App
    This application is designed to help you estimate causal effects from observational and A/B testing data using advanced statistical methods.
    
    ### 1. Data Preparation
    **Simulated Data**: If you don't have a dataset, select "Simulated Data" to explore the app's features with a generated dataset.
    
    **Upload Data**: Upload your own CSV file. Ensure your data contains:
    - **Treatment Column**: The variable indicating the intervention (e.g., `Feature_Adoption`, `Marketing_Campaign`).
    - **Outcome Column**: The metric you want to influence (e.g., `Account_Value`, `Conversion_Rate`).
    - **Confounders**: Variables that influence both treatment and outcome (e.g., `Customer_Segment`, `Region`).
    - **Data Preprocessing**:
        - **Auto-Convert Booleans**: Automatically converts TRUE/FALSE columns to 1/0.
        - **Missing Value Imputation**: Fill missing data using mean, median, or custom values.
        - **Winsorization**: Cap extreme outliers to reduce noise.
        - **Log Transformation**: Apply log transform to skewed variables.
        - **Standardization**: Scale variables to mean 0 and variance 1.
        - **Variable Bucketing**: Create categorical bins from numerical variables (e.g., Age Groups).
        - **Data Filtering**: Filter your dataset based on specific conditions (e.g., Region == 'North').
    - **Exploratory Analysis**:
        - **Data Preview**: View your raw and processed data.
        - **Correlation Matrix**: Analyze relationships between numeric variables.
        - **Chart Builder**: Create custom visualizations with **Aggregation** (Mean, Sum, Count, Median, Min, Max).
    - **Causal Analysis**:
        - **Binary Outcome Support**: Automatically detects binary outcomes (e.g., Conversion).
            - **OLS/Logit & DiD**: Automatically uses **Logit Model** (Odds Ratio).
            - **Other Methods**: Automatically adapts to use Classifiers.
        - **Average Treatment Effect (ATE)**: The overall impact of the intervention.
            - **For binary outcomes using Logit, ATE represents the relative odds of success (Odds Ratio or OR) which is converted to the absolute change in probability for ease of interpretation.
        - **Heterogeneous Treatment Effects (HTE)**: Analyze how the treatment effect varies by user characteristics (e.g., "Does the feature work better for mobile users?").
        - **Refutation Tests**: Validate your results (Placebo Treatment, Random Common Cause). *Note: Not available for Manual DiD or Logit models.*
        - **Optional Bootstrapping**: Enable/Disable bootstrapping for faster performance when Standard Errors are not needed.
     
    ### 2. Step-by-Step Causal Analysis
    #### Step 1: Define Causal Graph
    Select your Treatment, Outcome, and Confounders. This tells the app what relationship to measure and what to control for.
    
    #### Step 2: Choose Estimation Method
    | Method | When to use? |
    | :--- | :--- |
    | **OLS/Logit** | When you have a randomized controlled trial (RCT) or want simple regression adjustment. |
    | **Diff-in-Differences (DiD)** | When you have data over time (Pre/Post) and a Control group. Requires a `Time Period` column. |
    | **Propensity Score Matching** | To create a synthetic control group by matching similar users. |
    | **Inverse Propensity Weighting (IPTW)** | Weight observations by inverse probability of treatment to create a pseudo-population. |
    | **Double Machine Learning (LinearDML)** | For high-dimensional controls where you want a linear treatment effect. |
    | **Causal Forest (DML)** | For estimating non-linear Heterogeneous Treatment Effects (HTE). |
    | **Meta-Learners (S/T-Learner)** | For estimating Heterogeneous Treatment Effects (HTE) using Machine Learning. |
    
    #### Step 3: Interpret Results
    - **Average Treatment Effect (ATE)**: The overall impact of the intervention, with standard error and confidence interval.
    - **Heterogeneity Analysis**: (Now part of the Estimation section)
        - **Linear Methods**: Look for the "Interaction Coefficient". A significant p-value (< 0.05) means the effect varies by that feature.
        - **CATE Methods**: Look for the "Effect Modification" table. It shows which features drive the differences in individual effects.
    
    #### Step 4: Refutation
    - **Random Common Cause**: Checks stability by adding a random confounder.
    - **Placebo Treatment**: Checks validity by replacing treatment with a placebo (should be 0).
        
    ### 3. Export Data and Script
    - **View Generated Script**: Preview the full Python code directly in the app.
    - **Download Python Script**: Get a standalone Python file containing the analysis. You can run this locally to reproduce the results or integrate it into your pipeline.
    
    ### 4. Version History

    """)
    
    if history:
        # Display Latest Version
        st.markdown(f"**Latest Version:** {latest_version['Version']} ({latest_version['Release Date']})")
        st.info(f"**Release Note:** {latest_version['Release Note']}")
        
        # Display History Table
        with st.expander("See Full History"):
            history_df = pd.DataFrame(history)
            st.table(history_df)
    else:
        st.warning("No version history found.")

# Analysis Block
if st.session_state.get('analysis_run', False):
    with tab_causal: # Ensure results render in the Causal Tab

        with st.container(): # Main results container 
            # Ideally we dedent the whole block, but to minimize diff noise let's just remove the check.
            # Actually, let's just remove the if/else and dedent.
            pass

            st.divider()
            st.header("Causal Analysis Pipeline")
        
            # --- Step 1: Model ---
            st.subheader("1. Causal Model")
            st.markdown("**Methodology:** Structural Causal Model (SCM)")
            st.markdown("We define a Directed Acyclic Graph (DAG) $G = (V, E)$ where:")
            st.markdown(f"- $V$: Variables including Treatment (`{treatment}`), Outcome (`{outcome}`), and Confounders.")
            st.markdown("- $E$: Causal edges representing direct effects.")
            st.markdown("Assumption: **Causal Markov Assumption** (each variable is independent of its non-descendants given its parents).")
        
            # Only use confounders as effect modifiers for HTE-capable ML methods
            # For OLS/Logit, we want a simple adjustment without interaction terms in the main ATE model.
            if estimation_method in ["Double Machine Learning (LinearDML)", "Causal Forest (DML)", "Meta-Learner: S-Learner", "Meta-Learner: T-Learner"]:
                modifiers = confounders
            else:
                modifiers = []

            with st.spinner("Building Causal Graph..."):
                model = CausalModel(
                    data=df,
                    treatment=treatment,
                    outcome=outcome,
                    common_causes=confounders,
                    instruments=None,
                    effect_modifiers=modifiers
                )
        
            st.success("Model built successfully!")
            st.markdown("**Assumptions:**")
            st.write(f"Treatment: `{treatment}` causes Outcome: `{outcome}`")
            st.write(f"Confounders: `{', '.join(confounders)}` affect both.")
        

        
            # Visualize Graph (Optional - simplistic view)
            # st.graphviz_chart(model.view_model()) # Requires graphviz installed on system

            # --- Step 2: Identify ---
            st.subheader("2. Identification")
            st.markdown("**Methodology:** Backdoor Criterion")
            st.markdown("We aim to identify the causal effect $P(Y|do(T))$ from observational data $P(Y, T, X)$.")
            st.markdown("If a set of variables $X$ satisfies the Backdoor Criterion, we can use the **Adjustment Formula**:")
            st.latex(r"P(Y|do(T)) = \sum_X P(Y|T, X)P(X)")
        
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            st.write("Identified Estimand Type:", identified_estimand.estimand_type)
        
            # --- Step 3: Estimate (using EconML / DML) ---
            st.subheader(f"3. Estimation ({estimation_method})")
            with st.spinner(f"Estimating Causal Effect using {estimation_method}..."):
            
                use_logit = False # Initialize to avoid NameError in Refutation
                if estimation_method == "Double Machine Learning (LinearDML)":
                    st.markdown("#### Method: Double Machine Learning (DML)")
                    st.markdown("DML removes the effect of confounders ($X$) from both treatment ($T$) and outcome ($Y$) using ML models.")
                
                    st.markdown("**Step 1: Residualize Outcome**")
                    st.latex(r"Y_{res} = Y - E[Y|X]")
                
                    st.markdown("**Step 2: Residualize Treatment**")
                    st.latex(r"T_{res} = T - E[T|X]")
                
                    st.markdown("**Step 3: Estimate Causal Effect**")
                    st.latex(r"Y_{res} = \theta \cdot T_{res} + \epsilon")
                    st.caption("Where $\\theta$ is the Average Treatment Effect (ATE).")

                    # We use LinearDML from EconML
                    # It uses ML models to residualize treatment and outcome, then runs linear regression on residuals
                    
                    # For DML, even if the outcome is binary, we use a Regressor for model_y (LPM approach)
                    # because LinearDML expects a continuous residual or probability estimate, 
                    # and passing a Classifier can cause errors if EconML expects a Regressor interface.
                    model_y = RandomForestRegressor(n_jobs=-1, random_state=42)

                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.econml.dml.LinearDML",
                        method_params={
                            "init_params": {
                                "model_y": model_y,
                                "model_t": RandomForestClassifier(n_jobs=-1, random_state=42),
                                "discrete_treatment": True,
                                "linear_first_stages": False,
                                "cv": 3,
                                "random_state": 42
                            },
                            "fit_params": {}
                        }
                    )
                elif estimation_method == "Propensity Score Matching":
                    st.markdown("#### Method: Propensity Score Matching (PSM)")
                    if is_binary_outcome:
                        st.caption("ℹ️ **Binary Outcome**: Estimate represents Risk Difference (Difference in Proportions).")
                    st.markdown("PSM matches treated units with control units that have similar probability of receiving treatment.")
                
                    st.markdown("**Step 1: Estimate Propensity Score**")
                    st.latex(r"e(x) = P(T=1|X=x)")
                
                    st.markdown("**Step 2: Match Units**")
                    st.markdown("Find control unit $j$ for treated unit $i$ such that $e(x_i) \approx e(x_j)$.")
                
                    st.markdown("**Step 3: Estimate ATE**")
                    st.latex(r"ATE = \frac{1}{N} \sum_{i=1}^{N} (Y_i(1) - Y_{match(i)}(0))")
                
                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.propensity_score_matching"
                    )

                elif estimation_method == "Inverse Propensity Weighting (IPTW)":
                    st.markdown("#### Method: Inverse Propensity Weighting (IPTW)")
                    if is_binary_outcome:
                        st.caption("ℹ️ **Binary Outcome**: Estimate represents Risk Difference (Weighted Difference in Proportions).")
                    st.markdown("IPTW re-weights the data to create a pseudo-population where treatment is independent of confounders.")
                
                    st.markdown("**Step 1: Estimate Propensity Score**")
                    st.latex(r"e(x) = P(T=1|X=x)")
                
                    st.markdown("**Step 2: Calculate Weights**")
                    st.latex(r"w_i = \frac{T_i}{e(x_i)} + \frac{1-T_i}{1-e(x_i)}")
                
                    st.markdown("**Step 3: Estimate ATE**")
                    st.latex(r"ATE = \frac{1}{N} \sum_{i=1}^{N} w_i Y_i")
                
                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.propensity_score_weighting"
                    )

                elif "Meta-Learner" in estimation_method:
                    learner_type = estimation_method.split(": ")[1]
                    st.markdown(f"#### Method: {learner_type}")
                
                    if learner_type == "S-Learner":
                        st.markdown("S-Learner (Single Learner) treats treatment as a feature in a single ML model.")
                        st.latex(r"f(X, T) \approx Y")
                        st.latex(r"ATE = E[f(X, 1) - f(X, 0)]")
                        method_name = "backdoor.econml.metalearners.SLearner"
                        
                        if is_binary_outcome:
                            overall_model = RandomForestClassifier(n_jobs=-1, random_state=42)
                        else:
                            overall_model = RandomForestRegressor(n_jobs=-1, random_state=42)
                            
                        init_params = {"overall_model": overall_model}
                    else: # T-Learner
                        st.markdown("T-Learner (Two Learners) fits separate models for treated and control groups.")
                        st.latex(r"\mu_1(X) \approx E[Y|T=1, X], \quad \mu_0(X) \approx E[Y|T=0, X]")
                        st.latex(r"ATE = E[\mu_1(X) - \mu_0(X)]")
                        method_name = "backdoor.econml.metalearners.TLearner"
                        
                        if is_binary_outcome:
                            models = RandomForestClassifier(n_jobs=-1, random_state=42)
                        else:
                            models = RandomForestRegressor(n_jobs=-1, random_state=42)
                            
                        init_params = {"models": models}

                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name=method_name,
                        method_params={
                            "init_params": init_params,
                            "fit_params": {}
                        }
                    )

                elif estimation_method == "Causal Forest (DML)":
                    st.markdown("#### Method: Causal Forest (DML)")
                    st.markdown("Causal Forests extend Random Forests to estimate heterogeneous treatment effects (CATE) using an honest splitting criterion.")
                    st.latex(r"\hat{\tau}(x) = \frac{\sum \alpha_i(x) (Y_i - \hat{m}(X_i)) (T_i - \hat{e}(X_i))}{\sum \alpha_i(x) (T_i - \hat{e}(X_i))^2}")
                
                    # Use Regressor for model_y (LPM) to avoid errors with binary outcomes
                    model_y = RandomForestRegressor(n_jobs=-1, random_state=42)

                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.econml.dml.CausalForestDML",
                        method_params={
                            "init_params": {
                                "model_y": model_y,
                                "model_t": RandomForestClassifier(n_jobs=-1, random_state=42),
                                "discrete_treatment": True,
                                "random_state": 42
                            },
                            "fit_params": {}
                        }
                    )

                elif estimation_method == "Instrumental Variables (IV)":
                    st.markdown("#### Method: Instrumental Variables (IV)")
                    # Removed IV support
                    st.error("Instrumental Variables (IV) method is not supported in this version.")
                    st.stop()
                
                elif estimation_method == "Difference-in-Differences (DiD)":
                    st.markdown("#### Method: Difference-in-Differences (DiD)")
                    
                    use_logit = False
                    if is_binary_outcome:
                        use_logit = True
                        st.caption("ℹ️ **Binary Outcome**: Using **Logit Model** (Logistic Regression) to estimate **Odds Ratio**.")
                    
                    if not time_period:
                        st.error("Please select a Time Period in the sidebar.")
                        st.stop()
                    
                    st.markdown("DiD compares the changes in outcomes over time between a treatment group and a control group.")
                    if use_logit:
                        st.latex(r"\text{Logit Model: } \log\left(\frac{P(Y=1)}{1-P(Y=1)}\right) = \beta_0 + \beta_1 T + \beta_2 \text{Time} + \beta_3 (T \times \text{Time}) + \dots")
                        st.latex(r"\text{Odds Ratio (Interaction)} = e^{\beta_3}")
                    else:
                        st.latex(r"ATE = (E[Y|T=1, Post] - E[Y|T=1, Pre]) - (E[Y|T=0, Post] - E[Y|T=0, Pre])")
                    
                    # Manual DiD Implementation (OLS or Logit)
                    # Create interaction term
                    df_did = df.copy()
                    df_did['DiD_Interaction'] = df_did[treatment] * df_did[time_period]
                    
                    # Define X (Treatment, Time, Interaction, Confounders)
                    X_cols = [treatment, time_period, 'DiD_Interaction']
                    if confounders:
                        X_cols.extend(confounders)
                    X_did = df_did[X_cols]
                    X_did = sm.add_constant(X_did)
                    y_did = df_did[outcome]
                    
                    if use_logit:
                        try:
                            did_model = sm.Logit(y_did, X_did).fit(disp=0)
                            did_coeff = did_model.params['DiD_Interaction']
                            did_pvalue = did_model.pvalues['DiD_Interaction']
                            did_conf_int = did_model.conf_int().loc['DiD_Interaction']
                            odds_ratio = np.exp(did_coeff)
                            
                            st.markdown(f"**Estimated Odds Ratio (Interaction): {odds_ratio:.4f}**")
                            st.markdown(f"**Log-Odds Coefficient:** {did_coeff:.4f}")
                            st.markdown(f"**P-value:** {did_pvalue:.4f}")
                            st.markdown(f"**95% CI (Log-Odds):** [{did_conf_int[0]:.4f}, {did_conf_int[1]:.4f}]")
                            
                            # Convert OR to Risk Difference (Approximate)
                            # Baseline Risk (Control Group in Post Period - or overall Control)
                            # Let's use the overall Control group mean as a reference baseline
                            baseline_risk = df_did[df_did[treatment] == 0][outcome].mean()
                            if 0 < baseline_risk < 1:
                                def or_to_rd(odds_ratio, p0):
                                    return (odds_ratio * p0 / (1 - p0 + (odds_ratio * p0))) - p0
                                
                                implied_risk_diff = or_to_rd(odds_ratio, baseline_risk)
                                
                                # Convert CI
                                or_lower = np.exp(did_conf_int[0])
                                or_upper = np.exp(did_conf_int[1])
                                rd_lower = or_to_rd(or_lower, baseline_risk)
                                rd_upper = or_to_rd(or_upper, baseline_risk)
                                
                                st.markdown(f"**Implied Risk Difference (at Baseline Risk {baseline_risk:.2%}):** {implied_risk_diff:+.2%}")
                                st.markdown(f"**Implied RD 95% CI:** [{rd_lower:+.2%}, {rd_upper:+.2%}]")
                                st.caption("Note: Implied Risk Difference is calculated based on the observed control group mean.")
                            
                            # Create a dummy estimate object for compatibility (optional, or just skip standard display)
                            estimate = type('obj', (object,), {'value': did_coeff, 'estimator': type('obj', (object,), {'__str__': lambda self: "Logistic Regression (DiD)"})()})
                        except Exception as e:
                            st.error(f"Logit Model Failed: {e}")
                            st.stop()
                        
                    else:
                        try:
                            did_model = sm.OLS(y_did, X_did).fit()
                            if 'DiD_Interaction' in did_model.params:
                                did_estimate = did_model.params['DiD_Interaction']
                                st.markdown(f"**Estimated ATE (Interaction Coefficient): {did_estimate:.4f}**")
                                st.write(did_model.summary())
                                estimate = type('obj', (object,), {'value': did_estimate})
                            else:
                                st.error("DiD Interaction term dropped (likely due to collinearity). Check your data.")
                                st.stop()
                        except Exception as e:
                            st.error(f"LPM (OLS) Failed: {e}")
                            st.stop()
                    
                    # We skip standard DoWhy estimation for DiD because we did it manually
                    # But we need 'estimate' object for Refutation if we were to run it (Refutation disabled for DiD)
                    
                    # Let's use statsmodels for the DiD specific regression to get the p-value and summary.
                    import statsmodels.api as sm
                    


                elif estimation_method == "OLS/Logit":
                    st.markdown("#### Method: OLS/Logit")
                    
                    use_logit = False
                    if is_binary_outcome:
                        use_logit = True
                        st.caption("ℹ️ **Binary Outcome**: Using **Logit Model** (Logistic Regression) to estimate **Odds Ratio**.")
                    
                    st.markdown("Simple comparison of average outcomes between treatment and control groups.")
                    
                    if use_logit:
                        st.latex(r"\text{Odds Ratio} = \frac{P(Y=1|T=1)/P(Y=0|T=1)}{P(Y=1|T=0)/P(Y=0|T=0)}")
                        
                        try:
                            # Manual Logit Implementation for A/B
                            X_ab_cols = [treatment]
                            if confounders:
                                X_ab_cols.extend(confounders)
                            X_ab = df[X_ab_cols]
                            X_ab = sm.add_constant(X_ab)
                            y_ab = df[outcome]
                            
                            ab_model = sm.Logit(y_ab, X_ab).fit(disp=0)
                            ab_coeff = ab_model.params[treatment]
                            ab_pvalue = ab_model.pvalues[treatment]
                            ab_conf_int = ab_model.conf_int().loc[treatment]
                            odds_ratio = np.exp(ab_coeff)
                            
                            st.markdown(f"**Estimated Odds Ratio: {odds_ratio:.4f}**")
                            st.markdown(f"**Log-Odds Coefficient:** {ab_coeff:.4f}")
                            st.markdown(f"**P-value:** {ab_pvalue:.4f}")
                            st.markdown(f"**95% CI (Log-Odds):** [{ab_conf_int[0]:.4f}, {ab_conf_int[1]:.4f}]")
                            
                            # Convert OR to Risk Difference
                            # Baseline Risk (Control Group)
                            baseline_risk = df[df[treatment] == 0][outcome].mean()
                            if 0 < baseline_risk < 1:
                                def or_to_rd(odds_ratio, p0):
                                    return (odds_ratio * p0 / (1 - p0 + (odds_ratio * p0))) - p0
                                    
                                implied_risk_diff = or_to_rd(odds_ratio, baseline_risk)
                                
                                # Convert CI
                                or_lower = np.exp(ab_conf_int[0])
                                or_upper = np.exp(ab_conf_int[1])
                                rd_lower = or_to_rd(or_lower, baseline_risk)
                                rd_upper = or_to_rd(or_upper, baseline_risk)
                                
                                st.markdown(f"**Implied Risk Difference (at Baseline Risk {baseline_risk:.2%}):** {implied_risk_diff:+.2%}")
                                st.markdown(f"**Implied RD 95% CI:** [{rd_lower:+.2%}, {rd_upper:+.2%}]")
                                st.caption("Note: Implied Risk Difference is calculated based on the observed control group mean.")
                            
                            estimate = type('obj', (object,), {'value': ab_coeff})
                        except Exception as e:
                            st.error(f"Logit Model Failed: {e}")
                            st.stop()
                        
                    else:
                        st.latex(r"ATE = E[Y|T=1] - E[Y|T=0]")
                        try:
                            estimate = model.estimate_effect(
                                identified_estimand,
                                method_name="backdoor.linear_regression",
                                test_significance=True
                            )
                            
                            # Extract and Display Detailed Stats
                            try:
                                # DoWhy's LinearRegressionEstimator uses statsmodels internally
                                # Accessing the internal model to get summary stats
                                if hasattr(estimate.estimator, 'model'):
                                    sm_model = estimate.estimator.model
                                    # The treatment coefficient name might vary, usually it's the treatment name
                                    # But DoWhy might rename it. Let's check params.
                                    # Actually, estimate.value is the coefficient.
                                    # We can try to find the corresponding index or name.
                                    
                                    # Simpler approach: Use the test_significance results if available
                                    # But DoWhy's test_significance printout is just a print.
                                    
                                    # Let's try to show the full summary which is very informative
                                    st.markdown("**Regression Results:**")
                                    st.write(sm_model.summary())
                                    
                                    # Explicitly show SE and CI for Treatment
                                    # We need to identify the treatment variable name in the model
                                    # DoWhy usually keeps it as is or adds prefix.
                                    # Let's look at pvalues index
                                    treat_var = treatment
                                    if treat_var in sm_model.pvalues:
                                        se = sm_model.bse[treat_var]
                                        pval = sm_model.pvalues[treat_var]
                                        ci = sm_model.conf_int().loc[treat_var]
                                        
                                        st.markdown(f"**Standard Error:** {se:.4f}")
                                        st.markdown(f"**P-value:** {pval:.4f}")
                                        st.markdown(f"**95% CI:** [{ci[0]:.4f}, {ci[1]:.4f}]")
                            except Exception as e:
                                st.warning(f"Could not extract detailed stats: {e}")
                        except Exception as e:
                            st.error(f"LPM Estimation Failed: {e}")
                            st.stop()
                    
                    if confounders:
                        st.info("Note: Confounders included in regression (CUPED / Variance Reduction).")
                    else:
                        st.info("Note: Simple Difference in Means (Unadjusted).")
                    

            
            
            ate = estimate.value
        
            # --- Extract Standard Error & CI ---
            se = None
            ci = None
        
            try:
                # 1. Linear Regression (DiD approximation)
                if estimation_method == "Difference-in-Differences (DiD)":
                    if hasattr(estimate, 'estimator') and hasattr(estimate.estimator, 'model'):
                        model_res = estimate.estimator.model
                        if hasattr(model_res, 'bse'):
                            # Try to find the treatment variable in params
                            # Case 1: Exact match
                            if treatment in model_res.params.index:
                                se = model_res.bse[treatment]
                                conf_int = model_res.conf_int().loc[treatment]
                                ci = (conf_int[0], conf_int[1])
                            # Case 2: Contains match (e.g. T[T.1])
                            else:
                                for param_name in model_res.params.index:
                                    if treatment in param_name:
                                        se = model_res.bse[param_name]
                                        conf_int = model_res.conf_int().loc[param_name]
                                        ci = (conf_int[0], conf_int[1])
                                        break
                        
                            # Case 3: Generic name 'x1' (common in some DoWhy versions/inputs)
                            # We assume x1 is treatment if it's the first non-const variable and we haven't found it yet
                            if se is None and 'x1' in model_res.params.index:
                                 se = model_res.bse['x1']
                                 conf_int = model_res.conf_int().loc['x1']
                                 ci = (conf_int[0], conf_int[1])
            
                # 2. Other Methods (Check generic attributes)
                if se is None:
                    if hasattr(estimate, 'stderr'):
                        se = estimate.stderr
                
                    if hasattr(estimate, 'get_confidence_intervals'):
                        try:
                            ci_res = estimate.get_confidence_intervals(confidence_level=0.95)
                            if ci_res is not None:
                                # Handle different return formats (tuple, array, etc.)
                                if isinstance(ci_res, (list, tuple, np.ndarray)) and len(ci_res) == 2:
                                    ci = (float(ci_res[0]), float(ci_res[1]))
                        except Exception:
                            pass # CI extraction failed
                        
            except Exception as e:
                st.warning(f"Could not extract Standard Error: {e}")

            # --- Bootstrapping for SE ---
            # Bootstrapping is now default and configured in sidebar
        # --- 4. Bootstrapping (if enabled) ---
            se = None
            ci_lower = None
            ci_upper = None
            
            if n_iterations > 0:
                bootstrap_estimates = []
                progress_bar = st.progress(0)
            
                with st.spinner(f"Running {n_iterations} bootstrap iterations..."):
                    for i in range(n_iterations):
                        # Resample with replacement
                        df_resampled = df.sample(frac=1, replace=True, random_state=i) # Use i as seed for reproducibility of the set
                    
                        # Re-define model on resampled data
                        # Note: We must re-instantiate CausalModel to avoid state leakage
                        # Use same modifier logic as main model
                        if estimation_method in ["Double Machine Learning (LinearDML)", "Causal Forest (DML)", "Meta-Learner: S-Learner", "Meta-Learner: T-Learner"]:
                            modifiers_boot = confounders
                        else:
                            modifiers_boot = []
    
                        model_boot = CausalModel(
                            data=df_resampled,
                            treatment=treatment,
                            outcome=outcome,
                            common_causes=confounders,
                            instruments=None,
                            effect_modifiers=modifiers_boot
                        )
                    
                        identified_estimand_boot = model_boot.identify_effect(proceed_when_unidentifiable=True)
                    
                        # Re-estimate
                        # We need to use the exact same method and params
                        # This duplication is a bit verbose but necessary to ensure same config
                        try:
                            if estimation_method == "Double Machine Learning (LinearDML)":
                                est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name="backdoor.econml.dml.LinearDML",
                                    method_params={
                                        "init_params": {
                                            "model_y": RandomForestRegressor(random_state=42),
                                            "model_t": RandomForestClassifier(random_state=42),
                                            "discrete_treatment": True,
                                            "random_state": 42
                                        },
                                        "fit_params": {}
                                    }
                                )
                            elif estimation_method == "Propensity Score Matching":
                                    est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name="backdoor.propensity_score_matching"
                                )
                            elif estimation_method == "Inverse Propensity Weighting (IPTW)":
                                    est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name="backdoor.propensity_score_weighting"
                                )
                            elif "Meta-Learner" in estimation_method:
                                learner_type = estimation_method.split(": ")[1]
                                if learner_type == "S-Learner":
                                    method_name = "backdoor.econml.metalearners.SLearner"
                                    init_params = {"overall_model": RandomForestRegressor(random_state=42)}
                                else:
                                    method_name = "backdoor.econml.metalearners.TLearner"
                                    init_params = {"models": RandomForestRegressor(random_state=42)}
                            
                                est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name=method_name,
                                    method_params={
                                        "init_params": init_params,
                                        "fit_params": {}
                                    }
                                )
                            elif estimation_method == "Causal Forest (DML)":
                                    est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name="backdoor.econml.dml.CausalForestDML",
                                    method_params={
                                        "init_params": {
                                            "model_y": RandomForestRegressor(random_state=42),
                                            "model_t": RandomForestClassifier(random_state=42),
                                            "discrete_treatment": True,
                                            "random_state": 42
                                        },
                                        "fit_params": {}
                                    }
                                )
                            elif estimation_method == "Instrumental Variables (IV)":
                                    # Removed IV support
                                    continue # Skip this iteration if IV is selected
                            elif estimation_method == "Difference-in-Differences (DiD)":
                                    est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name="backdoor.linear_regression",
                                    test_significance=False # Speed up
                                )
                            elif estimation_method == "OLS/Logit":
                                    est_boot = model_boot.estimate_effect(
                                    identified_estimand_boot,
                                    method_name="backdoor.linear_regression",
                                    test_significance=False # Speed up
                                )
                                    # For Logit, we might need to extract the coefficient differently if we want OR
                                    # But for SE calculation, we usually want SE of the coefficient (log-odds)
                                    # est_boot.value should be the coefficient.
    
                        
                            bootstrap_estimates.append(est_boot.value)
                    
                        except Exception:
                            pass # Skip failed iterations
                    
                        progress_bar.progress((i + 1) / n_iterations)
            
                if len(bootstrap_estimates) > 0:
                    se = np.std(bootstrap_estimates)
                    ci_lower = np.percentile(bootstrap_estimates, 2.5)
                    ci_upper = np.percentile(bootstrap_estimates, 97.5)
                    st.success(f"Bootstrapping complete. Used {len(bootstrap_estimates)} successful iterations.")
                else:
                    st.error("Bootstrapping failed for all iterations.")
            else:
                st.info("Bootstrapping disabled. Standard Errors and Confidence Intervals will not be calculated.")

            # Display Metrics
            col_ate, col_se = st.columns(2)
            with col_ate:
                st.metric(label="Average Treatment Effect (ATE)", value=f"{ate:.2f}")
            with col_se:
                if se is not None:
                    st.metric(label="Standard Error (SE)", value=f"{se:.2f}")
                else:
                    st.metric(label="Standard Error (SE)", value="N/A", help="SE not available for this method/configuration.")
        
            if ci is not None:
                st.caption(f"**95% Confidence Interval:** [{ci[0]:.2f}, {ci[1]:.2f}]")
                st.caption("(Computed via Bootstrapping)")
        
            st.info(
                f"**Interpretation:** On average, `{treatment}` increases `{outcome}` by **{ate:.2f}** "
                "after accounting for confounding variables."
            )

            # --- Heterogeneity Analysis (Moved from Step 5) ---
            st.divider()
            st.subheader("Heterogeneity Analysis")
            
            hte_feature = None
        
            # Check if method supports Heterogeneous Treatment Effects (CATE)
            cate_methods = [
                "Double Machine Learning (LinearDML)",
                "Meta-Learner: S-Learner",
                "Meta-Learner: T-Learner",
                "Causal Forest (DML)"
            ]
        
            if estimation_method in cate_methods:
                st.markdown("#### Individual Treatment Effects (ITE)")
                st.markdown("Distribution of causal effects across the population.")
            
                try:
                    # EconML estimators in DoWhy are wrapped. 
                    # We need to pass the effect modifiers (X) to predict ITE.
                    # For this simple app, we'll use the confounders as X.
                    X_test = df[confounders]
                
                    # Accessing the underlying EconML estimator can be tricky via DoWhy's unified API
                    # But estimate.estimator object usually exposes it.
                    # However, DoWhy's CausalEstimator might not directly expose 'effect' for all.
                    # We will try to use the `estimate.estimator.effect(X)` if available.
                
                    if hasattr(estimate.estimator, 'effect'):
                         ite = estimate.estimator.effect(X_test)
                    elif hasattr(estimate, 'estimator_instance') and hasattr(estimate.estimator_instance, 'effect'):
                         ite = estimate.estimator_instance.effect(X_test)
                    else:
                        # Fallback for some DoWhy/EconML versions
                        ite = None
                        st.warning("Could not extract ITEs from this estimator version.")

                    if ite is not None:
                        # Flatten if necessary
                        ite = ite.flatten()
                    
                        fig, ax = plt.subplots()
                        ax.hist(ite, bins=30, alpha=0.7, color='green')
                        ax.axvline(estimate.value, color='red', linestyle='--', label='ATE')
                        ax.set_title("Distribution of Individual Treatment Effects")
                        ax.set_xlabel("Treatment Effect")
                        ax.set_ylabel("Frequency")
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)
                    
                        # Add ITE to dataframe for download
                        df_results = df.copy()
                        df_results['Estimated_ITE'] = ite
                    
                        # Feature Importance (Causal Forest only)
                        if estimation_method == "Causal Forest (DML)":
                            st.markdown("#### Feature Importance")
                            # EconML CausalForest has feature_importances_
                            if hasattr(estimate.estimator, 'feature_importances_'):
                                importances = estimate.estimator.feature_importances_
                                feat_names = confounders
                            
                                fig, ax = plt.subplots()
                                y_pos = np.arange(len(feat_names))
                                ax.barh(y_pos, importances, align='center')
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels(feat_names)
                                ax.invert_yaxis()  # labels read top-to-bottom
                                ax.set_title("Feature Importance for Heterogeneity")
                                ax.set_title("Feature Importance for Heterogeneity")
                                st.pyplot(fig)
                                plt.close(fig)

                        # --- Universal HTE Summary ---
                        st.markdown("**Effect Modification**")
                        st.markdown("Analysis of how covariates influence the estimated Individual Treatment Effects (ITE).")
                        
                        # Allow heterogeneity analysis on any column (except treatment/outcome/time)
                        valid_covariates_cate = [c for c in df.columns if c not in [treatment, outcome, time_period, 'Treatment_Encoded']]
                        
                        if not valid_covariates_cate:
                            st.warning("No covariates available for heterogeneity analysis.")
                        else:
                            st.markdown("**Analyzing heterogeneity for all available features...**")
                            hte_results_cate = []
                            progress_bar_cate = st.progress(0)
                            
                            # Create a temporary dataframe for regression
                            df_ite = df.copy()
                            df_ite['ITE'] = ite
                            
                            for i, feature in enumerate(valid_covariates_cate):
                                try:
                                    # Simple Linear Regression: ITE ~ Feature
                                    X_feat = sm.add_constant(df_ite[feature])
                                    y_feat = df_ite['ITE']
                                    
                                    model_feat = sm.OLS(y_feat, X_feat).fit()
                                    
                                    coef = model_feat.params[feature]
                                    pval = model_feat.pvalues[feature]
                                    
                                    hte_results_cate.append({
                                        "Feature": feature,
                                        "Effect Modification (Slope)": coef,
                                        "P-value": pval,
                                        "Significant (p<0.05)": "Yes" if pval < 0.05 else "No"
                                    })
                                except Exception as e:
                                    continue
                                
                                progress_bar_cate.progress((i + 1) / len(valid_covariates_cate))
                            
                            progress_bar_cate.empty()
                            
                            if hte_results_cate:
                                hte_df_cate = pd.DataFrame(hte_results_cate).sort_values("P-value")
                                st.dataframe(hte_df_cate.style.format({
                                    "Effect Modification (Slope)": "{:.4f}",
                                    "P-value": "{:.4f}"
                                }))
                                
                                # Highlight significant findings
                                sig_features_cate = hte_df_cate[hte_df_cate["P-value"] < 0.05]
                                if not sig_features_cate.empty:
                                    best_feat_cate = sig_features_cate.iloc[0]["Feature"]
                                    st.success(f"Found significant heterogeneity! The treatment effect is most strongly modified by **{best_feat_cate}**.")
                                else:
                                    st.info("No significant heterogeneity found across the analyzed features.")
                            else:
                                st.warning("Could not compute heterogeneity for any feature.")
                        

                except Exception as e:
                    st.error(f"Error calculating ITEs: {e}")
                    df_results = df.copy()

            elif estimation_method in ["OLS/Logit", "Difference-in-Differences (DiD)"]:
                st.markdown("Analyze how the treatment effect varies across different subgroups.")
                
                analyze_hte = True # Always enabled by default
                
                if analyze_hte:
                    # Allow heterogeneity analysis on any column (except treatment/outcome/time)
                    valid_covariates = [c for c in df.columns if c not in [treatment, outcome, time_period, 'Treatment_Encoded', 'DiD_Interaction', 'DiD_Interaction', 'HTE_Interaction']]
                    
                    if not valid_covariates:
                        st.warning("No covariates available for heterogeneity analysis.")
                        hte_feature = None
                        hte_results = []
                    else:
                        st.markdown("**Analyzing heterogeneity for all available features...**")
                        hte_results = []
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        
                        for i, feature in enumerate(valid_covariates):
                            try:
                                if estimation_method == "OLS/Logit":
                                    # Model: Y ~ T + X + T*X + Confounders
                                    df['HTE_Interaction'] = df[treatment] * df[feature]
                                    X_hte = df[[treatment, feature, 'HTE_Interaction']]
                                    other_confounders = [c for c in confounders if c != feature]
                                    if other_confounders:
                                        X_hte = pd.concat([X_hte, df[other_confounders]], axis=1)
                                    X_hte = sm.add_constant(X_hte)
                                    y_hte = df[outcome]
                                    hte_model = sm.OLS(y_hte, X_hte).fit()
                                    
                                    coef = hte_model.params['HTE_Interaction']
                                    pval = hte_model.pvalues['HTE_Interaction']
                                    
                                elif estimation_method == "Difference-in-Differences (DiD)":
                                    # Model: Y ~ T + Post + T*Post + X + T*X + Post*X + T*Post*X + Confounders
                                    df['T_X'] = df[treatment] * df[feature]
                                    df['Post_X'] = df[time_period] * df[feature]
                                    df['Triple_Interaction'] = df['DiD_Interaction'] * df[feature]
                                    
                                    X_hte = df[[treatment, time_period, 'DiD_Interaction', feature, 'T_X', 'Post_X', 'Triple_Interaction']]
                                    other_confounders = [c for c in confounders if c != feature]
                                    if other_confounders:
                                        X_hte = pd.concat([X_hte, df[other_confounders]], axis=1)
                                    X_hte = sm.add_constant(X_hte)
                                    y_hte = df[outcome]
                                    hte_model = sm.OLS(y_hte, X_hte).fit()
                                    
                                    coef = hte_model.params['Triple_Interaction']
                                    pval = hte_model.pvalues['Triple_Interaction']
                                
                                hte_results.append({
                                    "Feature": feature,
                                    "Interaction Coefficient": coef,
                                    "P-value": pval,
                                    "Significant (p<0.05)": "Yes" if pval < 0.05 else "No"
                                })
                            except Exception as e:
                                # Skip if error (e.g. collinearity)
                                continue
                            
                            progress_bar.progress((i + 1) / len(valid_covariates))
                        
                        progress_bar.empty()
                        
                        if hte_results:
                            hte_df = pd.DataFrame(hte_results).sort_values("P-value")
                            st.dataframe(hte_df.style.format({
                                "Interaction Coefficient": "{:.4f}",
                                "P-value": "{:.4f}"
                            }))
                            
                            # Highlight significant findings
                            sig_features = hte_df[hte_df["P-value"] < 0.05]
                            if not sig_features.empty:
                                best_feat = sig_features.iloc[0]["Feature"]
                                st.success(f"Found significant heterogeneity! The treatment effect varies most significantly by **{best_feat}**.")
                                hte_feature = best_feat # Set hte_feature for script export
                            else:
                                st.info("No significant heterogeneity found across the analyzed features.")
                                hte_feature = None # No single best feature
                        else:
                            st.warning("Could not compute heterogeneity for any feature.")

                df_results = df.copy()

            else:
                st.info(f"Individual Treatment Effects are not directly available for {estimation_method} in this view.")
                df_results = df.copy()

            csv = df_results.to_csv(index=False).encode('utf-8')


            # --- Step 4: Refute ---
            # Refutation (Step 4)
            st.subheader("4. Refutation")
            
            if estimation_method == "Difference-in-Differences (DiD)" or (is_binary_outcome and use_logit):
                st.warning("Refutation tests are not currently supported for this method/configuration (Manual Implementation).")
            else:
                st.markdown("### Methodologies")
                
                st.markdown("**1. Random Common Cause Test**")
                st.markdown("We add a random variable $W_{random}$ as a common cause to the dataset. Since $W_{random}$ is independent of the true process, the new estimate should not change significantly.")
                st.latex(r"ATE_{new} \approx ATE_{original}")
                
                st.markdown("**2. Placebo Treatment Refuter**")
                st.markdown("We replace the true treatment variable $T$ with an independent random variable $T_{placebo}$. Since the placebo treatment is random, it should have no effect on the outcome.")
                st.latex(r"ATE_{placebo} \approx 0")

                try:
                    with st.spinner("Running Refutation Tests..."):
                        # 1. Random Common Cause
                        refute_results_rcc = model.refute_estimate(
                            identified_estimand,
                            estimate,
                            method_name="random_common_cause"
                        )
                        
                        # 2. Placebo Treatment
                        refute_results_placebo = model.refute_estimate(
                            identified_estimand,
                            estimate,
                            method_name="placebo_treatment_refuter",
                            placebo_type="permute"
                        )
                    
                    # Display Results: Random Common Cause
                    st.write("**Test 1: Add Random Common Cause**")
                    st.write(f"Original Effect: {refute_results_rcc.estimated_effect:.2f}")
                    st.write(f"New Effect: {refute_results_rcc.new_effect:.2f}")
                    st.write(f"P-value: {refute_results_rcc.refutation_result['p_value']:.2f}")
                    
                    if refute_results_rcc.refutation_result['p_value'] > 0.05:
                         st.success("✅ Random Common Cause: Passed (Estimate is stable).")
                    else:
                         st.warning("⚠️ Random Common Cause: Warning (Estimate might be sensitive).")

                    st.divider()

                    # Display Results: Placebo Treatment
                    st.write("**Test 2: Placebo Treatment**")
                    st.markdown("Replaces the treatment with a random variable. The new effect should be close to 0.")
                    st.write(f"Original Effect: {refute_results_placebo.estimated_effect:.2f}")
                    st.write(f"New Effect (Placebo): {refute_results_placebo.new_effect:.2f}")
                    st.write(f"P-value: {refute_results_placebo.refutation_result['p_value']:.2f}")
                    
                    # For Placebo, we want the new effect to be close to 0, so p-value should ideally be > 0.05 
                    # (null hypothesis: effect is 0). 
                    # DoWhy's p-value here tests if the new effect is significantly different from 0.
                    # Wait, DoWhy's p-value interpretation depends on the test.
                    # For placebo, we want the effect to be insignificant (p > 0.05).
                    
                    if refute_results_placebo.refutation_result['p_value'] > 0.05:
                        st.success("✅ Placebo Treatment: Passed (Effect is indistinguishable from 0).")
                    else:
                        st.warning("⚠️ Placebo Treatment: Warning (Placebo effect is significant).")

                except Exception as e:
                    st.error(f"Refutation failed: {e}")



            # --- Step 5: Export Data and Script ---
            st.subheader("5. Export Data and Script")
            
            st.download_button(
                label="Download Data with Results as CSV",
                data=csv,
                file_name='causal_analysis_results.csv',
                mime='text/csv',
            )
        


            # Generate the script
            # We need to pass all current state variables
            # Note: Some variables like 'percentile' are only defined if winsorize_enable is True
            # We'll use defaults or current values.
        

            # Debug: Check if variables are available
            # st.write(f"Debug: data_source={data_source}")
            # st.write(f"Debug: control_val={control_val if 'control_val' in locals() else 'Not Found'}")
            
            try:
                script = generate_script(
                    data_source=data_source,
                    treatment=treatment,
                    outcome=outcome,
                    confounders=confounders,
                    time_period=time_period,
                    estimation_method=estimation_method,
                    impute_enable=impute_enable,
                    num_impute_method=num_impute_method if impute_enable else None,
                    num_custom_val=num_custom_val if impute_enable and num_impute_method == "Custom Value" else 0.0,
                    cat_impute_method=cat_impute_method if impute_enable else None,
                    cat_custom_val=cat_custom_val if impute_enable and cat_impute_method == "Custom Value" else "Missing",
                    winsorize_enable=winsorize_enable,
                    winsorize_cols=winsorize_cols if winsorize_enable else [],
                    percentile=percentile if winsorize_enable else 0.05,
                    log_transform_cols=log_transform_cols,
                    standardize_cols=standardize_cols,
                    n_iterations=n_iterations,
                    control_val=control_val if 'control_val' in locals() else None,
                    treat_val=treat_val if 'treat_val' in locals() else None,
                    hte_features=valid_covariates if 'valid_covariates' in locals() else (valid_covariates_cate if 'valid_covariates_cate' in locals() else None),
                    use_logit=use_logit,
                    bucketing_ops=st.session_state.bucketing_ops,
                    filtering_ops=st.session_state.filtering_ops
                )
            
                st.download_button(
                    label="Download Python Script",
                    data=script,
                    file_name="causal_analysis.py",
                    mime="text/x-python"
                )

                with st.expander("View Generated Python Script"):
                    st.code(script, language='python')
            except Exception as e:
                st.error(f"Error generating script: {e}")
