import textwrap
import numpy as np

def generate_script(data_source, treatment, outcome, confounders, time_period, estimation_method, 
                    impute_enable, num_impute_method, num_custom_val, cat_impute_method, cat_custom_val,
                    winsorize_enable, winsorize_cols, percentile,
                    log_transform_cols, standardize_cols, n_iterations):
    
    script = f"""import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from econml.dml import LinearDML, CausalForestDML
from econml.metalearners import SLearner, TLearner
import matplotlib.pyplot as plt

# --- 1. Load Data ---
"""
    if data_source == "Simulated Data":
        script += """
def simulate_data(n_samples=1000):
    np.random.seed(42)
    customer_segment = np.random.binomial(1, 0.3, n_samples)
    historical_usage = np.random.normal(50, 15, n_samples) + (customer_segment * 20)
    marketing_nudge = np.random.binomial(1, 0.5, n_samples)
    quarter = np.random.binomial(1, 0.5, n_samples)
    prob_adoption = 1 / (1 + np.exp(-( -2 + 0.5 * customer_segment + 0.05 * historical_usage + 1.5 * marketing_nudge)))
    feature_adoption = np.random.binomial(1, prob_adoption, n_samples)
    account_value = (200 + 500 * feature_adoption + 1000 * customer_segment + 10 * historical_usage + 50 * quarter + np.random.normal(0, 50, n_samples))
    
    df = pd.DataFrame({
        'Customer_Segment': customer_segment,
        'Historical_Usage': historical_usage,
        'Marketing_Nudge': marketing_nudge,
        'Quarter': quarter,
        'Feature_Adoption': feature_adoption,
        'Account_Value': account_value
    })
    return df

df = simulate_data()
print("Data Simulated Successfully")
"""
    else:
        script += """
# REPLACE 'your_dataset.csv' WITH THE PATH TO YOUR UPLOADED FILE
df = pd.read_csv('your_dataset.csv')
print("Data Loaded Successfully")
"""

    script += "\n# --- 2. Data Preprocessing ---\n"
    
    if impute_enable:
        script += f"""
# Imputation
num_cols = df.select_dtypes(include=[np.number]).columns
if len(num_cols) > 0:
    if "{num_impute_method}" == "Mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif "{num_impute_method}" == "Median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif "{num_impute_method}" == "Zero":
        df[num_cols] = df[num_cols].fillna(0)
    elif "{num_impute_method}" == "Custom Value":
        df[num_cols] = df[num_cols].fillna({num_custom_val})

cat_cols = df.select_dtypes(exclude=[np.number]).columns
if len(cat_cols) > 0:
    if "{cat_impute_method}" == "Mode":
        for col in cat_cols:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
    elif "{cat_impute_method}" == "Missing Indicator":
        df[cat_cols] = df[cat_cols].fillna("Missing")
    elif "{cat_impute_method}" == "Custom Value":
        df[cat_cols] = df[cat_cols].fillna("{cat_custom_val}")
print("Missing values imputed.")
"""

    if winsorize_enable and winsorize_cols:
        script += f"""
# Winsorization
winsorize_cols = {winsorize_cols}
percentile = {percentile}
for col in winsorize_cols:
    lower = df[col].quantile(percentile)
    upper = df[col].quantile(1 - percentile)
    df[col] = df[col].clip(lower=lower, upper=upper)
print(f"Winsorization applied to {{winsorize_cols}}")
"""

    if log_transform_cols:
        script += f"""
# Log Transformation
log_cols = {log_transform_cols}
for col in log_cols:
    if (df[col] < 0).any():
        print(f"Skipping {{col}} due to negative values")
    else:
        df[col] = np.log1p(df[col])
print("Log transformation applied.")
"""

    if standardize_cols:
        script += f"""
# Standardization
scaler = StandardScaler()
std_cols = {standardize_cols}
df[std_cols] = scaler.fit_transform(df[std_cols])
print("Standardization applied.")
"""

    script += f"""
# --- 3. Causal Model ---
treatment = '{treatment}'
outcome = '{outcome}'
confounders = {confounders}
instrument = None
effect_modifiers = confounders # Using confounders as effect modifiers for heterogeneity

model = CausalModel(
    data=df,
    treatment=treatment,
    outcome=outcome,
    common_causes=confounders,
    instruments=None,
    effect_modifiers=effect_modifiers
)

# --- 4. Identify Effect ---
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print("Effect Identified")

# --- 5. Estimate Effect ---
estimation_method = "{estimation_method}"
print(f"Estimating effect using {{estimation_method}}...")

"""
    script += "def estimate_causal_effect(model, identified_estimand):\n"
    script += "    estimate = None\n"

    # Logic for estimation methods (simplified mapping from app)
    if estimation_method == "Double Machine Learning (LinearDML)":
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='backdoor.econml.dml.LinearDML')\n"
    elif estimation_method == "Propensity Score Matching":
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='backdoor.propensity_score_matching')\n"
    elif estimation_method == "Inverse Propensity Weighting (IPTW)":
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='backdoor.propensity_score_weighting')\n"
    elif "Meta-Learner" in estimation_method:
        learner = "SLearner" if "S-Learner" in estimation_method else "TLearner"
        method_name = f"backdoor.econml.metalearners.{learner}"
        
        if learner == "SLearner":
            init_params_str = '{"overall_model": RandomForestRegressor(random_state=42)}'
        else:
            init_params_str = '{"models": RandomForestRegressor(random_state=42)}'
        
        script += "    method_name = '" + method_name + "'\n"
        script += "    init_params = " + init_params_str + "\n"
        script += "    estimate = model.estimate_effect(identified_estimand, method_name=method_name, method_params=dict(init_params=init_params, fit_params=dict()))\n"
    elif estimation_method == "Causal Forest (DML)":
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='backdoor.econml.dml.CausalForestDML', method_params=dict(init_params=dict(model_y=RandomForestRegressor(random_state=42), model_t=RandomForestClassifier(random_state=42), discrete_treatment=True, random_state=42), fit_params=dict()))\n"
    elif estimation_method == "Instrumental Variables (IV)":
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='iv.instrumental_variable')\n"
    elif estimation_method == "Difference-in-Differences (DiD)":
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression', test_significance=True)\n"
    
    script += "    return estimate\n"

    script += "estimate = estimate_causal_effect(model, identified_estimand)\n"
    script += "print(f'Average Treatment Effect (ATE): {estimate.value}')\n"
    script += "\n"
    script += "# --- 6. Refutation ---\n"
    script += "print('Running Refutation...')\n"
    script += "refute_results = model.refute_estimate(\n"
    script += "    identified_estimand,\n"
    script += "    estimate,\n"
    script += "    method_name='random_common_cause'\n"
    script += ")\n"
    script += "print(refute_results)\n"
    script += "\n"
    script += "# --- 7. Bootstrapping ---\n"
    script += f"print(f'Running {n_iterations} bootstrap iterations...')\n"
    script += "bootstrap_estimates = []\n"
    script += f"for i in range({n_iterations}):\n"
    script += "    try:\n"
    script += "        df_resampled = df.sample(frac=1, replace=True, random_state=i)\n"
    script += "        model_boot = CausalModel(\n"
    script += "            data=df_resampled,\n"
    script += f"            treatment='{treatment}',\n"
    script += f"            outcome='{outcome}',\n"
    script += f"            common_causes={confounders},\n"
    script += "            instruments=None,\n"
    script += f"            effect_modifiers={confounders}\n"
    script += "        )\n"
    script += "        identified_estimand_boot = model_boot.identify_effect(proceed_when_unidentifiable=True)\n"
    script += "        est_boot = estimate_causal_effect(model_boot, identified_estimand_boot)\n"
    script += "        bootstrap_estimates.append(est_boot.value)\n"
    script += "    except Exception:\n"
    script += "        pass\n"
    script += "\n"
    script += "if bootstrap_estimates:\n"
    script += "    se = np.std(bootstrap_estimates)\n"
    script += "    ci = (np.percentile(bootstrap_estimates, 2.5), np.percentile(bootstrap_estimates, 97.5))\n"
    script += "    print(f'Standard Error (SE): {se:.2f}')\n"
    script += "    print(f'95% Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]')\n"
    script += "else:\n"
    script += "    print('Bootstrapping failed.')\n"
    return script
