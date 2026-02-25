import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Streamlit UI
st.set_page_config(page_title="Monte Carlo Simulator", layout="wide")
st.title("Monte Carlo Retirement Simulator")
st.markdown("""**Description:**  
Traditional retirement calculators rely on static, normally distributed returns, which dangerously underestimate the impact of \"Black Swan\" market crashes and sequence-of-returns risk. This project utilizes a Monte Carlo decumulation engine to stress-test retirement portfolios against extreme macroeconomic conditions. Built in Python and deployed via Streamlit, it allows users to accurately project portfolio survival probabilities using robust statistical frameworks, including fat-tailed distributions and dynamic behavioral guardrails. This project involves the use of stochastic lifespans and copulas between market returns and inflation rates, which provides a step up on free prediction models.""")
st.divider()
st.markdown("""**How To Use:**  
Input your specific personal, market, and macroeconomic assumptions including dynamic income streams, behavioral guardrails, and effective tax rates. Click the \"Run Simulation\" button to run thousands of simulated lifecycles. Review your portfolio's performance metrics to gauge success rate, median real-dollar balance, and worst-case scenario outcomes.""")

# User Inputs (Sidebar)
st.sidebar.header("Personal Assumptions")
starting_balance = st.sidebar.number_input("Starting Balance ($)", value=1000000, step=50000)
initial_withdrawal = st.sidebar.number_input("Annual Withdrawal ($)", value=40000, step=5000)
years_in_retirement = st.sidebar.slider("Years in Retirement", min_value=10, max_value=50, value=30)

st.sidebar.header("Additional Income Streams")
num_income_streams = st.sidebar.selectbox("Number of Income Streams", [0, 1, 2, 3, 4], index=0)
income_streams = []

for j in range(num_income_streams):
    with st.sidebar.expander(f"Income Stream {j+1}", expanded=(j==0)):
        income_amount = st.number_input(f"Annual Income ($ Real)", value=20000, step=5000, key=f"amt_{j}")
        income_start = st.slider(f"Starts in Year", 0, years_in_retirement, 5, key=f"start_{j}")
        income_end = st.slider(f"Ends in Year", income_start, years_in_retirement, years_in_retirement, key=f"end_{j}")

        income_streams.append({"amount": income_amount, "start": income_start, "end": income_end})

st.sidebar.header("Taxes")
effective_tax_rate = st.sidebar.slider("Effective Tax Rate (%)", 0.0, 50.0, 15.0, 1.0, help="The blended average percentage you pay in taxes (not your top marginal bracket).") / 100

st.sidebar.header("Behavioral Guardrails")
minimum_withdrawal = st.sidebar.number_input("Minimum Living Standard ($ Real)", value=20000, step=2000)
max_withdrawal = st.sidebar.slider("Guardrail Trigger Threshold (%)", 1.0, 15.0, 6.0, 0.5, help="Guyton-Klinger rule is typically ~5.5%") / 100
max_withdrawal_paycut = st.sidebar.slider("Paycut Severity (%)", 1.0, 25.0, 10.0, 1.0, help="The academic standard is a 10% paycut.") / 100

st.sidebar.header("Market Assumptions and Glide Path")
st.caption("Shift your protfolio allocation over time. Set Start and End to the same number for a static portfolio.")
col_start, col_end = st.sidebar.columns(2)
with col_start:
    start_mean = st.number_input("Start Mean (%)", value=8.0, step=0.5) / 100
    start_volatility = st.number_input("Start Volatility (%)", value=15.0, step=1.0) / 100
with col_end:
    end_mean = st.number_input("End Mean(%)", value = 5.0, step=0.5) / 100
    end_volatility = st.number_input("End Volatility (%)", value=8.0, step=1.0) / 100

degrees_of_freedom = st.sidebar.slider("Fat Tails (Degrees of Freedom)", 3, 20, 5, help="The US Market generally exhibits 4 or 5 degrees of freedom.")

st.sidebar.header("Inflation & Macro")
average_inflation = st.sidebar.slider("Average Inflation (%)", 0.0, 10.0, 3.0, 0.5, help="The long-term historical US average is ~3%") / 100
inflation_volatility = st.sidebar.slider("Inflation Volatility (%)", 0.0, 5.0, 1.0, 0.1) / 100
corr = st.sidebar.slider("Stock/Inflation Correlation", -1.0, 1.0, -0.3, 0.1)

num_simulations = st.sidebar.selectbox("Number of Simulations", [10000, 25000, 50000], index=1)

# Simulation Button
if st.button("Run Simulation", type="primary"):
    
    with st.spinner(f'Vectorizing and running {num_simulations} lifetimes...'):
        
        # 1. Copula & Market Setup
        cov_matrix = [[1.0, corr], [corr, 1.0]]
        
        mean_path = np.linspace(start_mean, end_mean, years_in_retirement)
        vol_path = np.linspace(start_volatility, end_volatility, years_in_retirement)
        scaling_factor_path = vol_path * np.sqrt((degrees_of_freedom - 2) / degrees_of_freedom)
        
        # 2. Calibration
        np.random.seed(500)
        calibration_simulation_count = 500000
        calibration_shock = np.random.standard_t(degrees_of_freedom, size=calibration_simulation_count)
        scaled_calibration_grid = calibration_shock[:, None] * scaling_factor_path
        expectation_denominator_path = np.mean(np.exp(scaled_calibration_grid), axis=0)
        log_mean = np.log((1 + mean_path) / expectation_denominator_path)
        np.random.seed(None)

        # 3. Vectorization
        total_steps = num_simulations * years_in_retirement
        z_all = np.random.multivariate_normal([0, 0], cov_matrix, size=total_steps)

        u_returns_all = stats.norm.cdf(z_all[:, 0])
        shocks_all = stats.t.ppf(u_returns_all, df=degrees_of_freedom)
        z_inflation_all = z_all[:, 1]

        shocks_grid = shocks_all.reshape(num_simulations, years_in_retirement)
        z_inflation_grid = z_inflation_all.reshape(num_simulations, years_in_retirement)

        # Validation
        copula_returns = np.exp(log_mean_path + (shocks_grid * scaling_factor_path)) - 1
        actual_arithmetic_mean = np.mean(copula_returns)
        target_average_mean = np.mean(mean_path)

        # 4. The Monte Carlo Loop
        successful_lifetimes = 0
        ending_balances = []
        all_paths = []

        for i in range(num_simulations):
            current_balance = starting_balance
            current_withdrawal = initial_withdrawal
            path = [starting_balance]

            for year in range(years_in_retirement):
                # Aggregate Income
                total_income_this_year = 0
                for stream in income_streams:
                    if stream["start"] <= year < stream["end"]:
                        total_income_this_year += stream["amount"]

                # Portfolio Withdrawal
                actual_portfolio_withdrawal = current_withdrawal - total_income_this_year
                
                if actual_portfolio_withdrawal < 0:
                    actual_portfolio_withdrawal = 0

                gross_portfolio_withdrawal = actual_portfolio_withdrawal / (1 - effective_tax_rate)
                
                # Remove adjusted amount from the market
                current_balance -= gross_portfolio_withdrawal

                # Bankruptcy Check
                if current_balance <= 0:
                    current_balance = 0
                    path.append(current_balance)
                    path.extend([0] * (years_in_retirement - year - 1))
                    break

                # Correlated Market Returns & Inflation
                shock = shocks_grid[i, year]
                z_inflation = z_inflation_grid[i, year]

                this_year_scaling_factor = scaling_factor_path[year]
                this_year_log_mean = log_mean_path[year]
                
                this_year_nominal_return = this_year_log_mean + (shock * this_year_scaling_factor)
                nominal_growth_multiplier = np.exp(this_year_nominal_return)
                inflation_rate = average_inflation + (z_inflation * inflation_volatility)

                # Real Returns (Fisher Equation)
                real_growth_multiplier = nominal_growth_multiplier / (1 + inflation_rate)
                current_balance *= real_growth_multiplier

                # Parameterized Dynamic Guardrails
                if (gross_portfolio_withdrawal / current_balance) > max_withdrawal:
                    current_withdrawal *= (1 - max_withdrawal_paycut)
                    if current_withdrawal < minimum_withdrawal:
                        current_withdrawal = minimum_withdrawal

                path.append(current_balance)

            all_paths.append(path)
            ending_balances.append(current_balance)
            
            if current_balance > 0:
                successful_lifetimes += 1

        # Calculation Metrics
        success_rate = (successful_lifetimes / num_simulations) * 100
        median_ending_balance = np.median(ending_balances)
        worst_case = np.min(ending_balances)
        best_case = np.max(ending_balances)

        # Display
        st.divider()
        st.success("Simulation Complete! All figures displayed in **Today's Purchasing Power (Real Dollars)**.")
        
        # Proof and Metrics
        st.info(f"**Validation Check:** Target Average Mean across Glide Path was {target_average_mean*100:.2f}%. Your simulated Copula Arithmetic Mean was **{actual_arithmetic_mean*100:.2f}%**.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Success Rate", f"{success_rate:.1f}%")
        col2.metric("Median Balance", f"${median_ending_balance:,.0f}")
        col3.metric("Worst Case", f"${worst_case:,.0f}")
        col4.metric("Best Case", f"${best_case:,.0f}")

        # Graphs
        st.divider()
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("#### Spaghetti Plot")
            fig_spag, ax_spag = plt.subplots(figsize=(8, 5))
            for i in range(min(150, num_simulations)):
                ax_spag.plot(all_paths[i], color='teal', alpha=0.15)

            all_paths_arr = np.array(all_paths)
            ax_spag.plot(np.percentile(all_paths_arr, 50, axis=0), color='black', linewidth=2, label='Median')
            ax_spag.plot(np.percentile(all_paths_arr, 10, axis=0), color='red', linewidth=2, linestyle='dashed', label='Bottom 10%')
            ax_spag.plot(np.percentile(all_paths_arr, 90, axis=0), color='green', linewidth=2, linestyle='dashed', label='Top 10%')

            ax_spag.set_xlabel('Years in Retirement')
            ax_spag.set_ylabel('Portfolio Balance ($ Real)')
            ax_spag.legend()
            st.pyplot(fig_spag)

        with col_chart2:
            st.markdown("#### Probability Distribution of Final Outcomes")
            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
            
            p10 = np.percentile(ending_balances, 10)
            p50 = np.percentile(ending_balances, 50)
            p90 = np.percentile(ending_balances, 90)
            p95 = np.percentile(ending_balances, 95) # Used to cut off the extreme right tail
            
            weights = np.ones_like(ending_balances) / len(ending_balances)
            
            bins = np.linspace(0, p95, 40)
            n, bins, patches = ax_hist.hist(ending_balances, bins=bins, weights=weights, color='cadetblue', edgecolor='white')
            
            if len(patches) > 0:
                patches[0].set_facecolor('salmon')

            ax_hist.axvline(p10, color='red', linestyle='dashed', linewidth=2, label=f'10th %tile: ${p10:,.0f}')
            ax_hist.axvline(p50, color='black', linestyle='dashed', linewidth=2, label=f'Median: ${p50:,.0f}')
            ax_hist.axvline(p90, color='green', linestyle='dashed', linewidth=2, label=f'90th %tile: ${p90:,.0f}')
            
            ax_hist.set_xlabel('Final Portfolio Balance ($ Real)')
            ax_hist.set_ylabel('Probability')
            ax_hist.set_xlim(left=0, right=p95)
            
            ax_hist.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            
            ax_hist.legend()
            st.pyplot(fig_hist)
