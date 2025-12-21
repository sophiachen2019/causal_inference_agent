

import numpy as np

def generate_script(data_source, treatment, outcome, confounders, time_period, estimation_method, 
                    impute_enable, num_impute_method, num_custom_val, cat_impute_method, cat_custom_val,
                    winsorize_enable, winsorize_cols, percentile,
                    log_transform_cols, standardize_cols, n_iterations,

                    control_val=None, treat_val=None, hte_features=None, use_logit=False, bucketing_ops=None, filtering_ops=None):
    
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
import statsmodels.api as sm

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
    
    # Outcome: Conversion (Binary)
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

df = simulate_data()
print("Data Simulated Successfully")
"""
    else:
        script += """
# REPLACE 'your_dataset.csv' WITH THE PATH TO YOUR UPLOADED FILE
df = pd.read_csv('your_dataset.csv')
print("Data Loaded Successfully")

"""

    script += """
# --- Auto-Convert Boolean to Dummy ---
def convert_bool_to_int(df):
    # 1. Actual boolean types
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"Converted boolean columns to integer: {', '.join(bool_cols)}")
    
    # 2. String "TRUE"/"FALSE" (case insensitive)
    obj_cols = df.select_dtypes(include=['object']).columns
    mapping = {'TRUE': 1, 'FALSE': 0, 'T': 1, 'F': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
    
    for col in obj_cols:
        try:
            series_upper = df[col].astype(str).str.upper()
            sample = series_upper.dropna().head(100)
            if not set(sample.unique()).issubset(mapping.keys()):
                continue
            
            unique_vals = set(series_upper.dropna().unique())
            if unique_vals.issubset(mapping.keys()):
                df[col] = series_upper.map(mapping).fillna(df[col])
                df[col] = pd.to_numeric(df[col], errors='ignore')
                print(f"Converted string boolean column to integer: {col}")
        except:
            pass
    return df

df = convert_bool_to_int(df)
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
df[{standardize_cols}] = scaler.fit_transform(df[{standardize_cols}])
print("Standardization applied to: {', '.join(standardize_cols)}")
"""

    if filtering_ops:
        script += "\n# Data Filtering\n"
        for op in filtering_ops:
            col = op['col']
            operator = op['op']
            val = op['val']
            
            # Handle string vs numeric value in code generation
            if isinstance(val, str):
                val_repr = f"'{val}'"
            else:
                val_repr = str(val)

            if operator == "==":
                script += f"df = df[df['{col}'] == {val_repr}]\n"
            elif operator == "!=":
                script += f"df = df[df['{col}'] != {val_repr}]\n"
            elif operator == ">":
                script += f"df = df[df['{col}'] > {val_repr}]\n"
            elif operator == "<":
                script += f"df = df[df['{col}'] < {val_repr}]\n"
            elif operator == ">=":
                script += f"df = df[df['{col}'] >= {val_repr}]\n"
            elif operator == "<=":
                script += f"df = df[df['{col}'] <= {val_repr}]\n"
            elif operator == "contains":
                script += f"df = df[df['{col}'].astype(str).str.contains({val_repr}, na=False)]\n"
            
            script += f"print(f\"Applied filter: {col} {operator} {val_repr}. Rows remaining: {{len(df)}}\")\n"

    if bucketing_ops:
        script += "\n# Variable Bucketing\n"
        for op in bucketing_ops:
            col = op['col']
            n_bins = op['n_bins']
            method = op['method']
            new_col = op['new_col']
            
            if method == 'cut':
                script += f"df['{new_col}'] = pd.cut(df['{col}'], bins={n_bins}, labels=False)\n"
            else:
                script += f"df['{new_col}'] = pd.qcut(df['{col}'], q={n_bins}, labels=False, duplicates='drop')\n"
            
            script += f"print(f\"Created bucketed column '{new_col}' from '{col}'\")\n"

    script += f"""
# --- 3. Causal Model ---
# --- 3. Causal Model ---
"""
    if treat_val is not None and control_val is not None:
        # Handle categorical treatment encoding in the script
        script += "df['Treatment_Encoded'] = np.nan\n"
        script += f"df.loc[df['{treatment}'] == {repr(control_val)}, 'Treatment_Encoded'] = 0\n"
        script += f"df.loc[df['{treatment}'] == {repr(treat_val)}, 'Treatment_Encoded'] = 1\n"
        script += "df = df.dropna(subset=['Treatment_Encoded'])\n"
        script += "treatment = 'Treatment_Encoded'\n"
    else:
        script += f"treatment = '{treatment}'\n"

    script += f"""
outcome = '{outcome}'
confounders = {confounders}
instrument = None
use_logit = {use_logit}

# Only use confounders as effect modifiers for HTE-capable ML methods
if "{estimation_method}" in ["Double Machine Learning (LinearDML)", "Causal Forest (DML)", "Meta-Learner: S-Learner", "Meta-Learner: T-Learner"]:
    effect_modifiers = confounders
else:
    effect_modifiers = []


# Check for Binary Outcome
is_binary_outcome = False
if df[outcome].nunique() == 2:
    is_binary_outcome = True
    print(f"Detected binary outcome: {{outcome}}. Using Classification models.")
"""



    script += """
model = CausalModel(
    data=df,
    treatment=treatment,
    outcome=outcome,
    common_causes=confounders,
    instruments=instrument
)

# --- 4. Identify Effect ---
"""
    script += "identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)\n"
        
    script += """print("Effect Identified")

# --- 5. Estimate Effect ---
"""
    script += f'estimation_method = "{estimation_method}"\n'
    script += 'print(f"Estimating effect using {estimation_method}...")\n\n'

    script += "def estimate_causal_effect(model, identified_estimand, test_significance=True):\n"
    script += "    estimate = None\n"

    # Logic for estimation methods (simplified mapping from app)
    if estimation_method == "Double Machine Learning (LinearDML)":
        script += "    # Always use Regressor for model_y (LPM) to avoid errors with binary outcomes in LinearDML\n"
        script += "    # because LinearDML expects a continuous residual or probability estimate,\n"
        script += "    # and passing a Classifier can cause errors if EconML expects a Regressor interface.\n"
        script += "    model_y = RandomForestRegressor(n_jobs=-1, random_state=42)\n"
        script += "\n"
        script += "    estimate = model.estimate_effect(\n"
        script += "        identified_estimand,\n"
        script += "        method_name=\"backdoor.econml.dml.LinearDML\",\n"
        script += "        method_params={\n"
        script += "            \"init_params\": {\n"
        script += "                \"model_y\": model_y,\n"
        script += "                \"model_t\": RandomForestClassifier(n_jobs=-1, random_state=42),\n"
        script += "                \"discrete_treatment\": True,\n"
        script += "                \"linear_first_stages\": False,\n"
        script += "                \"cv\": 3,\n"
        script += "                \"random_state\": 42\n"
        script += "            },\n"
        script += "            \"fit_params\": {}\n"
        script += "        }\n"
        script += "    )\n"
    elif estimation_method == "Propensity Score Matching":
        script += "    if is_binary_outcome:\n"
        script += "        print('Binary Outcome: Estimate represents Risk Difference (Difference in Proportions).')\n"
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='backdoor.propensity_score_matching')\n"
    elif estimation_method == "Inverse Propensity Weighting (IPTW)":
        script += "    if is_binary_outcome:\n"
        script += "        print('Binary Outcome: Estimate represents Risk Difference (Weighted Difference in Proportions).')\n"
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='backdoor.propensity_score_weighting')\n"
    elif "Meta-Learner" in estimation_method:
        learner = "SLearner" if "S-Learner" in estimation_method else "TLearner"
        method_name = f"backdoor.econml.metalearners.{learner}"
        
        if learner == "SLearner":
            script += "    if is_binary_outcome:\n"
            script += "        overall_model = RandomForestClassifier(n_jobs=-1, random_state=42)\n"
            script += "    else:\n"
            script += "        overall_model = RandomForestRegressor(n_jobs=-1, random_state=42)\n"
            script += "    init_params = {\"overall_model\": overall_model}\n"
        else: # T-Learner
            script += "    print(\"T-Learner (Two Learners) fits separate models for treated and control groups.\")\n"
            script += "    method_name = \"backdoor.econml.metalearners.TLearner\"\n"
            script += "    if is_binary_outcome:\n"
            script += "        models = RandomForestClassifier(n_jobs=-1, random_state=42)\n"
            script += "    else:\n"
            script += "        models = RandomForestRegressor(n_jobs=-1, random_state=42)\n"
            script += "    init_params = {\"models\": models}\n"
        
        script += "    estimate = model.estimate_effect(identified_estimand, method_name=method_name, method_params=dict(init_params=init_params, fit_params=dict()))\n"
    elif estimation_method == "Causal Forest (DML)":
        # Always use Regressor for model_y (LPM)
        script += "    model_y = RandomForestRegressor(n_jobs=-1, random_state=42)\n"
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='backdoor.econml.dml.CausalForestDML', method_params=dict(init_params=dict(model_y=model_y, model_t=RandomForestClassifier(n_jobs=-1, random_state=42), discrete_treatment=True, random_state=42), fit_params=dict()))\n"

    elif estimation_method == "Difference-in-Differences (DiD)":
        script += "    # ----------------------------------------------------------------\n"
        script += "    # Manual DiD Estimation\n"
        script += "    # We manually fit OLS with an interaction term (Treatment * Time)\n"
        script += "    # because standard DoWhy estimators do not automatically handle\n"
        script += "    # this specific DiD formulation.\n"
        script += "    # ----------------------------------------------------------------\n"
        script += "    if is_binary_outcome:\n"
        script += "        if use_logit:\n"
        script += "            print('Binary Outcome: Using Logit Model (Logistic Regression). Estimate represents Odds Ratio.')\n"
        script += "        else:\n"
        script += "            print('Binary Outcome: Using Linear Probability Model (LPM) approach. Estimate represents Risk Difference.')\n"
        script += "    data = model._data.copy()\n"
        script += f"    data['DiD_Interaction'] = data['{treatment}'] * data['{time_period}']\n"
        script += f"    X_did = data[['{treatment}', '{time_period}', 'DiD_Interaction']]\n"
        script += f"    if {confounders}:\n"
        script += f"        X_did = pd.concat([X_did, data[{confounders}]], axis=1)\n"
        script += "    X_did = sm.add_constant(X_did)\n"
        script += f"    y_did = data['{outcome}']\n"
        
        script += "    if use_logit and is_binary_outcome:\n"
        script += "        did_model = sm.Logit(y_did, X_did).fit(disp=0)\n"
        script += "        did_coeff = did_model.params['DiD_Interaction']\n"
        script += "        odds_ratio = np.exp(did_coeff)\n"
        script += "        print(did_model.summary())\n"
        script += "        print(f'Estimated Odds Ratio (Interaction): {odds_ratio:.4f}')\n"
        script += "        # Convert OR to Risk Difference\n"
        script += f"        baseline_risk = data[data['{treatment}'] == 0]['{outcome}'].mean()\n"
        script += "        if 0 < baseline_risk < 1:\n"
        script += "            def or_to_rd(or_val, p0):\n"
        script += "                return (or_val * p0 / (1 - p0 + (or_val * p0))) - p0\n"
        script += "            implied_risk_diff = or_to_rd(odds_ratio, baseline_risk)\n"
        script += "            # CI Conversion\n"
        script += "            did_conf_int = did_model.conf_int().loc['DiD_Interaction']\n"
        script += "            or_lower = np.exp(did_conf_int[0])\n"
        script += "            or_upper = np.exp(did_conf_int[1])\n"
        script += "            rd_lower = or_to_rd(or_lower, baseline_risk)\n"
        script += "            rd_upper = or_to_rd(or_upper, baseline_risk)\n"
        script += "            print(f'Implied Risk Difference (at Baseline Risk {baseline_risk:.2%}): {implied_risk_diff:+.2%}')\n"
        script += "            print(f'Implied RD 95% CI: [{rd_lower:+.2%}, {rd_upper:+.2%}]')\n"
        script += "        estimate = type('obj', (object,), {'value': did_coeff})\n"
        script += "    else:\n"
        script += "        did_model = sm.OLS(y_did, X_did).fit()\n"
        script += "        print(did_model.summary())\n"
        script += "        estimate = type('obj', (object,), {'value': did_model.params['DiD_Interaction']})\n"

    elif estimation_method == "OLS/Logit":
        script += "    if is_binary_outcome:\n"
        script += "        if use_logit:\n"
        script += "            print('Binary Outcome: Using Logit Model (Logistic Regression). Estimate represents Odds Ratio.')\n"
        script += "        else:\n"
        script += "            print('Binary Outcome: Using Linear Probability Model (LPM) approach. Estimate represents Risk Difference.')\n"
        
        script += "    if use_logit and is_binary_outcome:\n"
        script += f"        X_ab = df[['{treatment}'] + confounders]\n"
        script += "        X_ab = sm.add_constant(X_ab)\n"
        script += f"        y_ab = df['{outcome}']\n"
        script += "        ab_model = sm.Logit(y_ab, X_ab).fit(disp=0)\n"
        script += f"        ab_coeff = ab_model.params['{treatment}']\n"
        script += "        odds_ratio = np.exp(ab_coeff)\n"
        script += "        print(ab_model.summary())\n"
        script += "        print(f'Estimated Odds Ratio: {odds_ratio:.4f}')\n"
        script += "        # Convert OR to Risk Difference\n"
        script += f"        baseline_risk = df[df['{treatment}'] == 0]['{outcome}'].mean()\n"
        script += "        if 0 < baseline_risk < 1:\n"
        script += "            def or_to_rd(or_val, p0):\n"
        script += "                return (or_val * p0 / (1 - p0 + (or_val * p0))) - p0\n"
        script += "            implied_risk_diff = or_to_rd(odds_ratio, baseline_risk)\n"
        script += "            # CI Conversion\n"
        script += "            ab_conf_int = ab_model.conf_int().loc['{treatment}']\n"
        script += "            or_lower = np.exp(ab_conf_int[0])\n"
        script += "            or_upper = np.exp(ab_conf_int[1])\n"
        script += "            rd_lower = or_to_rd(or_lower, baseline_risk)\n"
        script += "            rd_upper = or_to_rd(or_upper, baseline_risk)\n"
        script += "            print(f'Implied Risk Difference (at Baseline Risk {baseline_risk:.2%}): {implied_risk_diff:+.2%}')\n"
        script += "            print(f'Implied RD 95% CI: [{rd_lower:+.2%}, {rd_upper:+.2%}]')\n"
        script += "        estimate = type('obj', (object,), {'value': ab_coeff})\n"
        script += "    else:\n"
        script += "        estimate = model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression', test_significance=test_significance)\n"
    
    script += "    return estimate\n"

    script += "estimate = estimate_causal_effect(model, identified_estimand)\n"
    script += "print(f'Average Treatment Effect (ATE): {estimate.value}')\n"
    script += "\n"
    script += "# --- 6. Refutation ---\n"
    # Skip Refutation for DiD and Logit models as they are manually implemented
    if estimation_method == "Difference-in-Differences (DiD)":
        script += "print('Refutation tests are not currently supported for Difference-in-Differences (Manual Implementation).')\n"
    elif estimation_method == "OLS/Logit":
        if use_logit:
            script += "print('Refutation tests are not currently supported for Logit Model (Manual Implementation).')\n"
        else:
            script += "print('Running Refutation...')\n"
            script += "refute_results = model.refute_estimate(identified_estimand, estimate, method_name='random_common_cause')\n"
            script += "print(refute_results)\n"
    else:
        script += "print('Running Refutation...')\n"
        script += "refute_results = model.refute_estimate(\n"
        script += "    identified_estimand,\n"
        script += "    estimate,\n"
        script += "    method_name='random_common_cause'\n"
        script += ")\n"
        script += "print(refute_results)\n"

    script += "\n"
    script += "# --- 7. Bootstrapping (if enabled) ---\n"
    script += "if n_iterations > 0:\n"
    script += f"    print(f'\\nRunning {n_iterations} bootstrap iterations...')\n"
    script += "    bootstrap_estimates = []\n"
    script += f"    for i in range({n_iterations}):\n"
    script += "        try:\n"
    script += "            df_resampled = df.sample(frac=1, replace=True, random_state=i)\n"
    script += "            model_boot = CausalModel(\n"
    script += "                data=df_resampled,\n"
    script += f"                treatment='{treatment}',\n"
    script += f"                outcome='{outcome}',\n"
    script += f"                common_causes={confounders},\n"
    script += "                instruments=instrument,\n"
    script += "                effect_modifiers=effect_modifiers\n"
    script += "            )\n"
    
    script += "            identified_estimand_boot = model_boot.identify_effect(proceed_when_unidentifiable=True)\n"
    script += "            est_boot = estimate_causal_effect(model_boot, identified_estimand_boot, test_significance=False)\n"
    script += "            bootstrap_estimates.append(est_boot.value)\n"
    script += "        except Exception:\n"
    script += "            pass\n"
    script += "\n"
    script += "    if bootstrap_estimates:\n"
    script += "        se = np.std(bootstrap_estimates)\n"
    script += "        ci = (np.percentile(bootstrap_estimates, 2.5), np.percentile(bootstrap_estimates, 97.5))\n"
    script += "        print(f'Standard Error (SE): {se:.2f}')\n"
    script += "        print(f'95% Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]')\n"
    script += "    else:\n"
    script += "        print('Bootstrapping failed.')\n"

    if hte_features:
        script += "\n"
        script += "# --- 8. Heterogeneity Analysis ---\n"
        script += "print('\\nRunning Heterogeneity Analysis for all features...')\n"
        script += "hte_results = []\n"
        
        # We need to pass the list of features as a string representation of a list
        script += f"features_to_analyze = {hte_features}\n"
        script += "for feature in features_to_analyze:\n"
        script += "    try:\n"
        
        if estimation_method == "OLS/Logit":
            script += "        # Model: Y ~ T + X + T*X + Confounders\n"
            script += "        df['HTE_Interaction'] = df[treatment] * df[feature]\n"
            script += "        X_hte = df[[treatment, feature, 'HTE_Interaction']]\n"
            script += f"        other_confounders = [c for c in {confounders} if c != feature]\n"
            script += "        if other_confounders:\n"
            script += "            X_hte = pd.concat([X_hte, df[other_confounders]], axis=1)\n"
            script += "        X_hte = sm.add_constant(X_hte)\n"
            script += "        y_hte = df[outcome]\n"
            script += "        hte_model = sm.OLS(y_hte, X_hte).fit()\n"
            script += "        coef = hte_model.params['HTE_Interaction']\n"
            script += "        pval = hte_model.pvalues['HTE_Interaction']\n"
            
        elif estimation_method == "Difference-in-Differences (DiD)":
            script += "        # Model: Y ~ T + Post + T*Post + X + T*X + Post*X + T*Post*X + Confounders\n"
            script += "        data = model._data.copy()\n"
            script += "        # Recalculate DiD_Interaction if missing (it's Treatment * Time)\n"
            script += f"        data['DiD_Interaction'] = data[treatment] * data['{time_period}']\n"
            script += "        data['T_X'] = data[treatment] * data[feature]\n"
            script += f"        data['Post_X'] = data['{time_period}'] * data[feature]\n"
            script += "        data['Triple_Interaction'] = data['DiD_Interaction'] * data[feature]\n"
            
            script += f"        X_hte = data[[treatment, '{time_period}', 'DiD_Interaction', feature, 'T_X', 'Post_X', 'Triple_Interaction']]\n"
            script += f"        other_confounders = [c for c in {confounders} if c != feature]\n"
            script += "        if other_confounders:\n"
            script += "            X_hte = pd.concat([X_hte, data[other_confounders]], axis=1)\n"
            
            script += "        X_hte = sm.add_constant(X_hte)\n"
            script += "        y_hte = data[outcome]\n"
            script += "        hte_model = sm.OLS(y_hte, X_hte).fit()\n"
            script += "        coef = hte_model.params['Triple_Interaction']\n"
            script += "        pval = hte_model.pvalues['Triple_Interaction']\n"

        elif estimation_method in ["Double Machine Learning (LinearDML)", "Meta-Learner: S-Learner", "Meta-Learner: T-Learner", "Causal Forest (DML)"]:
            script += "        # Universal HTE: Regress ITE on Feature\n"
            script += "        # Calculate ITE first\n"
            script += f"        X_test = df[{confounders}]\n"
            script += "        try:\n"
            script += "            if hasattr(estimate.estimator, 'effect'):\n"
            script += "                ite = estimate.estimator.effect(X_test)\n"
            script += "            elif hasattr(estimate, 'estimator_instance') and hasattr(estimate.estimator_instance, 'effect'):\n"
            script += "                ite = estimate.estimator_instance.effect(X_test)\n"
            script += "            else:\n"
            script += "                ite = None\n"
            script += "                print('Warning: Could not extract ITEs from estimator.')\n"
            script += "        except Exception as e:\n"
            script += "            ite = None\n"
            script += "            print(f'Error calculating ITE: {e}')\n"
            script += "        \n"
            script += "        if ite is not None:\n"
            script += "            df_ite = df.copy()\n"
            script += "            df_ite['ITE'] = ite.flatten()\n"
            script += "            \n"
            script += "            X_feat = sm.add_constant(df_ite[feature])\n"
            script += "            y_feat = df_ite['ITE']\n"
            script += "            model_feat = sm.OLS(y_feat, X_feat).fit()\n"
            script += "            coef = model_feat.params[feature]\n"
            script += "            pval = model_feat.pvalues[feature]\n"
            script += "        else:\n"
            script += "            raise ValueError('ITE not found for CATE method.')\n"

        script += "        hte_results.append({\n"
        script += "            'Feature': feature,\n"
        script += "            'Interaction Coefficient': coef,\n"
        script += "            'P-value': pval,\n"
        script += "            'Significant': 'Yes' if pval < 0.05 else 'No'\n"
        script += "        })\n"
        script += "    except Exception as e:\n"
        script += "        print(f'Skipping {feature} due to error: {e}')\n"
        script += "        continue\n"
        
        script += "\n"
        script += "if hte_results:\n"
        script += "    hte_df = pd.DataFrame(hte_results).sort_values('P-value')\n"
        script += "    print('\\n--- Heterogeneity Analysis Results ---')\n"
        script += "    print(hte_df)\n"
        script += "else:\n"
        script += "    print('No heterogeneity results computed.')\n"

    return script
