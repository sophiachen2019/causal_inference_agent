import streamlit as st
import textwrap
import pandas as pd
import numpy as np
import dowhy
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from econml.dml import LinearDML, CausalForestDML
from econml.metalearners import SLearner, TLearner
import matplotlib.pyplot as plt
from dowhy import CausalModel
import dowhy.datasets
from scipy import stats
import statsmodels.api as sm


# Import custom utils with explicit reload to ensure updates are picked up
import causal_utils
import importlib
importlib.reload(causal_utils)
from causal_utils import generate_script

# --- 1. Data Simulation ---
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
    
    df = pd.DataFrame({
        'Customer_Segment': customer_segment,
        'Historical_Usage': historical_usage,
        'Marketing_Nudge': marketing_nudge,
        'Quarter': quarter,
        'Feature_Adoption': feature_adoption,
        'Account_Value': account_value
    })
    
    # Enforce Data Types
    df['Customer_Segment'] = df['Customer_Segment'].astype(int)
    df['Historical_Usage'] = df['Historical_Usage'].astype(float)
    df['Marketing_Nudge'] = df['Marketing_Nudge'].astype(int)
    df['Quarter'] = df['Quarter'].astype(int)
    df['Feature_Adoption'] = df['Feature_Adoption'].astype(int)
    df['Account_Value'] = df['Account_Value'].astype(float)
    
    return df

# --- Streamlit UI ---
st.set_page_config(page_title="Causal Inference Agent", layout="wide")
st.title("🤖 Causal Inference Agent")
st.markdown("**Builder:** Sophia Chen | **Version:** v1 | **Email:** sophiachen2012@gmail.com | **Medium:** https://medium.com/@sophiachen2012")

# Load Data
# --- Tabs Setup ---
tab_eda, tab_causal = st.tabs(["Exploratory Analysis", "Causal Analysis"])

# ==========================================
# TAB 1: Exploratory Analysis
# ==========================================
with tab_eda:
    st.header("Exploratory Data Analysis")
    
    # --- Data Source ---
    st.subheader("1. Data Source")
    data_source = st.radio("Data Source", ["Simulated Data", "Upload CSV"], horizontal=True)
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Awaiting CSV upload. Using simulated data for preview.")
            df = simulate_data()
    else:
        df = simulate_data()

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
# TAB 2: Causal Analysis
# ==========================================
with tab_causal:
    st.header("Causal Analysis Configuration")
    
    # Ensure columns exist in df (handle case where upload might have different columns)
    def get_index(columns, default_name, default_idx):
        if default_name in columns:
            return list(columns).index(default_name)
        return default_idx if default_idx < len(columns) else 0

    estimation_method = st.selectbox(
        "Estimation Method",
        [
            "A/B Test (Difference in Means)",
            "Double Machine Learning (LinearDML)", 
            "Propensity Score Matching",
            "Inverse Propensity Weighting (IPTW)",
            "Meta-Learner: S-Learner",
            "Meta-Learner: T-Learner",
            "Causal Forest (DML)",
            "Difference-in-Differences (DiD)"
        ]
    )

    rd_running_variable = None
    rd_cutoff = 0.0
    rd_bandwidth = 0.0

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
    
    default_confounders = [c for c in ['Customer_Segment', 'Historical_Usage'] if c in df.columns]
    confounders = st.multiselect("Confounders (Common Causes)", df.columns, default=default_confounders)
    
    # Optional Inputs for Advanced Methods
    time_period = st.selectbox("Time Period (for DiD)", [None] + list(df.columns), index=0)
    
    st.markdown("#### Bootstrapping Settings")
    n_iterations = st.number_input("Bootstrap Iterations", min_value=10, max_value=500, value=50, step=10, help="Number of resampling iterations for SE estimation.")

    run_analysis = st.button("Run Causal Analysis", type="primary")

if run_analysis:
    with tab_causal: # Ensure results render in the Causal Tab

        if True: # Placeholder to maintain indentation level for now, or we can dedent. 
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
        
            with st.spinner("Building Causal Graph..."):
                model = CausalModel(
                    data=df,
                    treatment=treatment,
                    outcome=outcome,
                    common_causes=confounders,
                    instruments=None,
                    effect_modifiers=confounders
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
                
                    est = LinearDML(
                        model_y=RandomForestRegressor(random_state=42),
                        model_t=RandomForestClassifier(random_state=42),
                        discrete_treatment=True,
                        linear_first_stages=False,
                        cv=3,
                        random_state=42
                    )
                
                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.econml.dml.LinearDML",
                        method_params={
                            "init_params": {
                                "model_y": RandomForestRegressor(random_state=42),
                                "model_t": RandomForestClassifier(random_state=42),
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
                        init_params = {"overall_model": RandomForestRegressor(random_state=42)}
                    else: # T-Learner
                        st.markdown("T-Learner (Two Learners) fits separate models for treated and control groups.")
                        st.latex(r"\mu_1(X) \approx E[Y|T=1, X], \quad \mu_0(X) \approx E[Y|T=0, X]")
                        st.latex(r"ATE = E[\mu_1(X) - \mu_0(X)]")
                        method_name = "backdoor.econml.metalearners.TLearner"
                        init_params = {"models": RandomForestRegressor(random_state=42)}

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
                
                    estimate = model.estimate_effect(
                        identified_estimand,
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
                    st.markdown("#### Method: Instrumental Variables (IV)")
                    # Removed IV support
                    st.error("Instrumental Variables (IV) method is not supported in this version.")
                    st.stop()
                
                elif estimation_method == "Difference-in-Differences (DiD)":
                    st.markdown("#### Method: Difference-in-Differences (DiD)")
                    if not time_period:
                        st.error("Please select a Time Period in the sidebar.")
                        st.stop()

                    st.markdown("DiD compares the changes in outcomes over time between a treatment group and a control group.")
                    st.latex(r"ATE = (E[Y|T=1, Post] - E[Y|T=1, Pre]) - (E[Y|T=0, Post] - E[Y|T=0, Pre])")
                
                    # For DiD, we need the interaction term: Treatment * Time
                    # Y = a + b1*T + b2*Time + b3*(T*Time) + e
                    # The effect is b3.
                    
                    # Create interaction term
                    df['DiD_Interaction'] = df[treatment] * df[time_period]
                    
                    # Prepare features for Linear Regression
                    # We use sklearn or statsmodels. Let's use statsmodels for nice summary if available, 
                    # but we have sklearn imported. Let's use sklearn for consistency with other parts or simple OLS.
                    # Actually, we want standard errors. DoWhy's linear_regression gives SEs.
                    # We can trick DoWhy by passing the interaction as a "confounder" or "modifier"? No.
                    
                    # Let's use statsmodels for the DiD specific regression to get the p-value and summary.
                    import statsmodels.api as sm
                    
                    X_did = df[[treatment, time_period, 'DiD_Interaction']]
                    if confounders:
                        X_did = pd.concat([X_did, df[confounders]], axis=1)
                    
                    X_did = sm.add_constant(X_did)
                    y_did = df[outcome]
                    
                    did_model = sm.OLS(y_did, X_did).fit()
                    
                    did_estimate = did_model.params['DiD_Interaction']
                    
                    # Create a dummy estimate object to match the flow
                    estimate = type('obj', (object,), {
                        'value': did_estimate,
                        'params': did_model.params
                    })
                    
                    st.write(did_model.summary())
                    st.info(f"DiD Estimate (Interaction Term): {did_estimate:.4f}")
                    
                    # Skip the default DoWhy estimate for DiD as we did it manually
                    estimate.value = did_estimate # Ensure downstream usage works if any

                elif estimation_method == "A/B Test (Difference in Means)":
                    st.markdown("#### Method: A/B Test (Difference in Means)")
                    st.markdown("Simple comparison of average outcomes between treatment and control groups.")
                    st.latex(r"ATE = E[Y|T=1] - E[Y|T=0]")
                
                    # We use Linear Regression (OLS) Y ~ T (+ X)
                    # This provides ATE and standard errors.
                
                    estimate = model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.linear_regression",
                        test_significance=True
                    )
                
                    if confounders:
                        st.info("Note: Confounders included in regression (CUPED / Variance Reduction).")
                    else:
                        st.info("Note: Simple Difference in Means (Unadjusted).")
                    
                    # --- Covariate Balance Check ---
                    if confounders:
                        st.markdown("#### Covariate Balance (Randomization Check)")
                        balance_data = []
                        for col in confounders:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                # Calculate means
                                mean_control = df[df[treatment]==0][col].mean()
                                mean_treat = df[df[treatment]==1][col].mean()
                                diff = mean_treat - mean_control
                            
                                # T-test
                                try:
                                    t_stat, p_val = stats.ttest_ind(
                                        df[df[treatment]==1][col].dropna(), 
                                        df[df[treatment]==0][col].dropna(), 
                                        equal_var=False
                                    )
                                except Exception:
                                    p_val = np.nan
                                
                                balance_data.append({
                                    "Covariate": col,
                                    "Mean (Control)": mean_control,
                                    "Mean (Treatment)": mean_treat,
                                    "Diff": diff,
                                    "P-Value": p_val
                                })
                    
                        if balance_data:
                            st.dataframe(pd.DataFrame(balance_data).style.format({
                                "Mean (Control)": "{:.2f}",
                                "Mean (Treatment)": "{:.2f}",
                                "Diff": "{:.2f}",
                                "P-Value": "{:.3f}"
                            }))
                            st.caption("P-values < 0.05 indicate potential imbalance (randomization failure).")
            
            
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
        
            bootstrap_estimates = []
            progress_bar = st.progress(0)
        
            with st.spinner(f"Running {n_iterations} bootstrap iterations..."):
                for i in range(n_iterations):
                    # Resample with replacement
                    df_resampled = df.sample(frac=1, replace=True, random_state=i) # Use i as seed for reproducibility of the set
                
                    # Re-define model on resampled data
                    # Note: We must re-instantiate CausalModel to avoid state leakage
                    model_boot = CausalModel(
                        data=df_resampled,
                        treatment=treatment,
                        outcome=outcome,
                        common_causes=confounders,
                        instruments=None,
                        effect_modifiers=confounders
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
                        elif estimation_method == "A/B Test (Difference in Means)":
                                est_boot = model_boot.estimate_effect(
                                identified_estimand_boot,
                                method_name="backdoor.linear_regression",
                                test_significance=False # Speed up
                            )
                    
                        bootstrap_estimates.append(est_boot.value)
                
                    except Exception:
                        pass # Skip failed iterations
                
                    progress_bar.progress((i + 1) / n_iterations)
        
            if len(bootstrap_estimates) > 0:
                se = np.std(bootstrap_estimates)
                ci = (np.percentile(bootstrap_estimates, 2.5), np.percentile(bootstrap_estimates, 97.5))
                st.success(f"Bootstrapping complete. Used {len(bootstrap_estimates)} successful iterations.")
            else:
                st.error("Bootstrapping failed for all iterations.")

            # Display Metrics
            col_ate, col_se = st.columns(2)
            with col_ate:
                st.metric(label="Average Treatment Effect (ATE)", value=f"${ate:.2f}")
            with col_se:
                if se is not None:
                    st.metric(label="Standard Error (SE)", value=f"{se:.2f}")
                else:
                    st.metric(label="Standard Error (SE)", value="N/A", help="SE not available for this method/configuration.")
        
            if ci is not None:
                st.caption(f"**95% Confidence Interval:** [{ci[0]:.2f}, {ci[1]:.2f}]")
                st.caption("(Computed via Bootstrapping)")
        
            st.info(
                f"**Interpretation:** On average, `{treatment}` increases `{outcome}` by **${ate:.2f}** "
                "after accounting for confounding variables."
            )

            # --- Step 4: Refute ---
            # --- Step 4: Refute ---
            st.subheader("4. Refutation")
            
            if estimation_method == "Difference-in-Differences (DiD)":
                 st.warning("Refutation tests are not currently supported for the manual Difference-in-Differences implementation.")
            else:
                st.markdown("**Methodology:** Random Common Cause Test")
                st.markdown("We add a random variable $W_{random}$ as a common cause to the dataset.")
                st.markdown("Since $W_{random}$ is independent of the true process, the new estimate should not change significantly.")
                st.latex(r"Y_{new} = f(T, X, W_{random}) + \epsilon")
                st.markdown("Expected Result: $ATE_{new} \\approx ATE_{original}$")

                try:
                    with st.spinner("Running Refutation Tests..."):
                        refute_results = model.refute_estimate(
                            identified_estimand,
                            estimate,
                            method_name="random_common_cause"
                        )
                    
                    st.write("**Test: Add Random Common Cause**")
                    st.write(f"Original Effect: {refute_results.estimated_effect:.2f}")
                    st.write(f"New Effect: {refute_results.new_effect:.2f}")
                    st.write(f"P-value: {refute_results.refutation_result['p_value']:.2f}")
                
                    if refute_results.refutation_result['p_value'] > 0.05: # Simplistic check, usually we check if new effect is close to original
                         st.success("✅ Robustness Check Passed: The estimate is stable.")
                    else:
                         st.warning("⚠️ Robustness Check Warning: The estimate might be sensitive.")
                except Exception as e:
                    st.error(f"Refutation failed: {e}")

            # --- Step 5: Explore Results ---
            st.subheader("5. Explore Results")
        
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
                                st.pyplot(fig)
                                plt.close(fig)
                        
                except Exception as e:
                    st.error(f"Error calculating ITEs: {e}")
                    df_results = df.copy()

            else:
                st.info(f"Individual Treatment Effects are not directly available for {estimation_method} in this view.")
                df_results = df.copy()

            # Download Results
            st.markdown("#### Download Results")
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data with Results as CSV",
                data=csv,
                file_name='causal_analysis_results.csv',
                mime='text/csv',
            )

            # --- Export Analysis Script ---
            st.markdown("#### Export Analysis Script")
        


            # Generate the script
            # We need to pass all current state variables
            # Note: Some variables like 'percentile' are only defined if winsorize_enable is True
            # We'll use defaults or current values.
        
            # Helper to safely get variable or default
            def safe_get(var_name, default):
                return locals().get(var_name, default)

            analysis_script = generate_script(
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
                control_val=safe_get('control_val', None),
                treat_val=safe_get('treat_val', None)
            )
        
            st.download_button(
                label="Download Python Script",
                data=analysis_script,
                file_name='reproduce_analysis.py',
                mime='text/x-python',
            )

            # --- Decisioning ---
            st.divider()
            st.header("💡 Recommendation")
        
            if ate > 0:
                st.markdown(f"Based on the causal analysis, adopting **{treatment}** has a **positive** impact on **{outcome}**.\\n\\n**Action:**\\n- Roll out this feature to more customers.\\n- Invest in marketing campaigns to drive adoption.")
            else:
                st.markdown(f"Based on the causal analysis, adopting **{treatment}** has a **negligible or negative** impact on **{outcome}**.\\n\\n**Action:**\\n- Re-evaluate the value proposition.\\n- Do not prioritize broad rollout at this time.")
