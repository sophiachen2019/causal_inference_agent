
import numpy as np

def generate_script(data_source, treatment, outcome, confounders, time_period, estimation_method, 
                    impute_enable, num_impute_method, num_custom_val, cat_impute_method, cat_custom_val,
                    winsorize_enable, winsorize_cols, percentile,
                    log_transform_cols, standardize_cols, n_iterations,

                    control_val=None, treat_val=None, hte_features=None):
    
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
effect_modifiers = confounders # Using confounders as effect modifiers for heterogeneity
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

    elif estimation_method == "Difference-in-Differences (DiD)":
        script += "    # ----------------------------------------------------------------\n"
        script += "    # Manual DiD Estimation\n"
        script += "    # We manually fit OLS with an interaction term (Treatment * Time)\n"
        script += "    # because standard DoWhy estimators do not automatically handle\n"
        script += "    # this specific DiD formulation.\n"
        script += "    # ----------------------------------------------------------------\n"
        script += "    data = model._data.copy()\n"
        script += f"    data['DiD_Interaction'] = data['{treatment}'] * data['{time_period}']\n"
        script += f"    X_did = data[['{treatment}', '{time_period}', 'DiD_Interaction']]\n"
        script += f"    if {confounders}:\n"
        script += f"        X_did = pd.concat([X_did, data[{confounders}]], axis=1)\n"
        script += "    X_did = sm.add_constant(X_did)\n"
        script += f"    y_did = data['{outcome}']\n"
        script += "    did_model = sm.OLS(y_did, X_did).fit()\n"
        script += "    print(did_model.summary())\n"
        script += "    # Return dummy object with value\n"
        script += "    estimate = type('obj', (object,), {'value': did_model.params['DiD_Interaction']})\n"

    elif estimation_method == "A/B Test (Difference in Means)":
        script += "    estimate = model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression', test_significance=test_significance)\n"
    
    script += "    return estimate\n"

    script += "estimate = estimate_causal_effect(model, identified_estimand)\n"
    script += "print(f'Average Treatment Effect (ATE): {estimate.value}')\n"
    script += "\n"
    script += "# --- 6. Refutation ---\n"
    if estimation_method == "Difference-in-Differences (DiD)":
        script += "print('Skipping Refutation for Manual DiD')\n"
        script += "# Refutation skipped\n"
    else:
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
    script += "            instruments=instrument,\n"
    script += f"            effect_modifiers={confounders}\n"
    script += "        )\n"
    
    script += "        identified_estimand_boot = model_boot.identify_effect(proceed_when_unidentifiable=True)\n"
    script += "        est_boot = estimate_causal_effect(model_boot, identified_estimand_boot, test_significance=False)\n"
    script += "        bootstrap_estimates.append(est_boot.value)\n"
    script += "    except Exception:\n"
    script += "        pass\n"
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
        
        if estimation_method == "A/B Test (Difference in Means)":
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
