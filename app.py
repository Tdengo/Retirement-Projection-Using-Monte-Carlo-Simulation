import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Streamlit UI
st.set_page_config(page_title="Advanced Monte Carlo Simulator", layout="wide")
st.title("Monte Carlo Retirement Simulator")
st.markdown("Fully Vectorized Log-t Engine with Gaussian Copulas & Dynamic Guardrails.")

# User Inputs (Sidebar)
st.sidebar.header("Personal Assumptions")
starting_balance = st.sidebar.number_input("Starting Balance ($)", value=1000000, step=50000)
initial_withdrawal = st.sidebar.number_input("Annual Withdrawal ($)", value=40000, step=5000)
years_in_retirement = st.sidebar.slider("Years in Retirement", min_value=10, max_value=50, value=30)

st.sidebar.header("Additional Income Streams \n(Social Security, Pension, Revenue, etc.)")
num_income_streams = st.sidebar.selectbox("Number of Income Streams", [0, 1, 2, 3, 4], index=1)
income_streams = []

for j in range(num_income_streams):
    with st.sidebar.expander(f"Income Stream {j+1}", expanded=(j==0)):
        income_amount = st.number_input(f"Annual Income ($ Real)", value=20000, step=5000, key=f"amt_{j}")
        income_start = st.slider(f"Starts in Year", 0, years_in_retirement, 5, key=f"start_{j}")
        income_end = st.slider(f"Ends in Year", income_start, years_in_retirement, years_in_retirement, key=f"end_{j}")

        income_streams.append({"amount": income_amount, "start": income_start, "end": income_end})

st.sidebar.header("Behavioral Guardrails")
minimum_withdrawal = st.sidebar.number_input("Minimum Living Standard ($ Real)", value=20000, step=2000)
max_withdrawal = st.sidebar.slider("Guardrail Trigger Threshold (%) n\(Guyton-Klinger rule ~5.5%)", 1.0, 15.0, 6.0, 0.5) / 100
max_withdrawal_paycut = st.sidebar.slider("Paycut Severity (%) n\(Academic Standard ~10%)", 1.0, 25.0, 10.0, 1.0) / 100

st.sidebar.header("Market Assumptions")
target_arithmetic_mean = st.sidebar.slider("Target Arithmetic Mean (%) n\(Historical nominal return ~8%)", 1.0, 15.0, 7.0, 0.5) / 100
volatility = st.sidebar.slider("Volatility (%)", 5.0, 30.0, 15.0, 1.0) / 100
degrees_of_freedom = st.sidebar.slider("Fat Tails (Degrees of Freedom) n\(US Market generally exhibits 4, 5)", 3, 20, 5)

st.sidebar.header("Inflation & Macro")
average_inflation = st.sidebar.slider("Average Inflation (%) n\(Long-term historical average ~3%)", 0.0, 10.0, 3.0, 0.5) / 100
inflation_volatility = st.sidebar.slider("Inflation Volatility (%)", 0.0, 5.0, 1.0, 0.1) / 100
corr = st.sidebar.slider("Stock/Inflation Correlation", -1.0, 1.0, -0.3, 0.1)

num_simulations = st.sidebar.selectbox("Number of Simulations", [10000, 25000, 50000], index=1)

# Simulation Button
if st.button("Run Simulation", type="primary"):
    
    with st.spinner(f'Vectorizing and running {num_simulations} lifetimes...'):
        
        # 1. Copula & Market Setup
        cov_matrix = [[1.0, corr], [corr, 1.0]]
        scaling_factor = volatility * np.sqrt((degrees_of_freedom - 2) / degrees_of_freedom)

        # 2. Calibration
        np.random.seed(500)
        calibration_simulation_count = 500000
        calibration_shock = np.random.standard_t(degrees_of_freedom, size=calibration_simulation_count)
        scaled_calibration_shock = calibration_shock * scaling_factor
        expectation_denominator = np.mean(np.exp(scaled_calibration_shock))
        log_mean = np.log((1 + target_arithmetic_mean) / expectation_denominator)
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
        flat_shocks = shocks_grid.flatten()
        scaled_flat = flat_shocks * scaling_factor
        copula_returns = np.exp(log_mean + scaled_flat) - 1
        actual_arithmetic_mean = np.mean(copula_returns)

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

                # Remove adjusted amount from the market
                current_balance -= actual_portfolio_withdrawal

                # Bankruptcy Check
                if current_balance <= 0:
                    current_balance = 0
                    path.append(current_balance)
                    path.extend([0] * (years_in_retirement - year - 1))
                    break

                # Correlated Market Returns & Inflation
                shock = shocks_grid[i, year]
                z_inflation = z_inflation_grid[i, year]

                this_year_nominal_return = log_mean + (shock * scaling_factor)
                nominal_growth_multiplier = np.exp(this_year_nominal_return)
                inflation_rate = average_inflation + (z_inflation * inflation_volatility)

                # Real Returns (Fisher Equation)
                real_growth_multiplier = nominal_growth_multiplier / (1 + inflation_rate)
                current_balance *= real_growth_multiplier

                # Parameterized Dynamic Guardrails
                if (actual_portfolio_withdrawal / current_balance) > max_withdrawal:
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
        st.info(f"**Validation Check:** Target Arithmetic Mean was {target_arithmetic_mean*100:.2f}%. Your simulated Copula Arithmetic Mean was **{actual_arithmetic_mean*100:.2f}%**.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Success Rate", f"{success_rate:.1f}%")
        col2.metric("Median Balance", f"${median_ending_balance:,.0f}")
        col3.metric("Worst Case", f"${worst_case:,.0f}")
        col4.metric("Best Case", f"${best_case:,.0f}")

        # Graphs
        st.divider()
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("#### The Journey (Spaghetti Plot)")
            fig_spag, ax_spag = plt.subplots(figsize=(8, 5))
            for i in range(min(200, num_simulations)):
                ax_spag.plot(all_paths[i], color='teal', alpha=0.1)

            all_paths_arr = np.array(all_paths)
            ax_spag.plot(np.percentile(all_paths_arr, 50, axis=0), color='black', linewidth=2, label='Median')
            ax_spag.plot(np.percentile(all_paths_arr, 10, axis=0), color='red', linewidth=2, linestyle='dashed', label='Bottom 10%')
            ax_spag.plot(np.percentile(all_paths_arr, 90, axis=0), color='green', linewidth=2, linestyle='dashed', label='Top 10%')

            ax_spag.set_xlabel('Years in Retirement')
            ax_spag.set_ylabel('Portfolio Balance ($ Real)')
            ax_spag.legend()
            st.pyplot(fig_spag)

        with col_chart2:
            st.markdown("#### Final Outcomes (Distribution)")
            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
            ax_hist.hist(ending_balances, bins=50, color='cadetblue', edgecolor='black')
            ax_hist.axvline(median_ending_balance, color='mistyrose', linestyle='dashed', linewidth=2, label=f'Median: ${median_ending_balance:,.0f}')
            
            ax_hist.set_xlabel('Final Portfolio Balance ($ Real)')
            ax_hist.set_ylabel('Number of Occurrences')
            ax_hist.legend()

            st.pyplot(fig_hist)


