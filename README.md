https://retirement-projection-using-monte-carlo-simulation-tgmbdx2xnin.streamlit.app/

**Summary:** Traditional retirement calculators rely on static, normally distributed returns, which dangerously underestimate the impact of "Black Swan" market crashes and sequence-of-returns risk. This project utilizes a Monte Carlo decumulation engine to stress-test retirement portfolios against extreme macroeconomic conditions. Built entirely in Python and deployed via Streamlit, it allows users to accurately project portfolio survival probabilities using advanced statistical frameworks, including fat-tailed distributions and dynamic behavioral guardrails. This project involves the use of stochastic lifespans and copulas between market returns and inflation rates, which provides a step up on free prediction models. 

**Key Features**
-  **Log-t Distribution Modeling:** Replaces standard bell-curve returns with a Student's t-distribution to accurately model the "fat tails" of historical financial markets.
-  **Gaussian Copula Engine:** Correlates market returns with inflation rates, allowing for the simulation of devastating "stagflation" environments where portfolios crash while the cost of living spikes.
-  **Dynamic Spending Guardrails:** Implements programmable withdrawal rules (e.g., Guyton-Klinger) that automatically reduce spending during market stress while maintaining a hard, inflation-adjusted survival floor.
-  **Real-Dollar Purchasing Power:** Utilizes the Fisher Equation to automatically convert all simulation outcomes from nominal dollars into real purchasing power for accurate lifestyle planning.
-  **High-Performance Vectorization:** Pre-computes 1.5 million correlated random variables using NumPy array broadcasting, reducing simulation time for 50,000 lifetimes.
