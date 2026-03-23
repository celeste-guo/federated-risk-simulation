# Simulating Decentralized Risk Models in Finance

### Overview
As a student bridging an academic background in Finance and Economics with a quantitative focus on Data Science, my current objective is to translate financial risk modeling concepts into Python-based data architectures.

Financial institutions face strict regulatory constraints (e.g., GDPR and bank secrecy regulations) that prevent the pooling of client-level data. This creates a fundamental barrier to training robust, cross-institutional machine learning models. This project serves as a simplified pedagogical simulation to understand the baseline mechanics of overcoming this barrier via decentralized collaborative learning.

### Project Architecture
The simulation models a decentralized learning environment across three distinct institutional profiles:
1. **Private Wealth Office:** High-net-worth clientele with conservative, balanced portfolios.
2. **Standard Retail Bank:** Middle-income consumers with average savings and low market participation.
3. **FinTech NeoBroker:** A younger demographic utilizing highly aggressive, volatile equity strategies.

The pipeline executes the following methodology:
* **Data Generation:** Engineering synthetic client portfolios (incorporating Age, Assets Under Management, and Equity Ratios) where the underlying probability of client attrition is derived from a simplified economic baseline.
* **Localized Optimization:** Executing logistic regression via gradient descent entirely within each institution's isolated data silo.
* **Federated Aggregation:** Aggregating the locally optimized mathematical parameters—weighted by sample size—to construct a global predictive model without exposing any underlying raw client data.

### Technical Implementation
To ensure a rigorous understanding of the underlying calculus and linear algebra before relying on high-level machine learning frameworks (e.g., Scikit-Learn or PyTorch), this simulation is intentionally built from scratch. Feature standardization, cross-entropy optimization, and gradient derivation are implemented entirely utilizing base NumPy arrays and Pandas data structures.
