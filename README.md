# Market Microstructure Analysis of INFY: A Multivariate Hawkes Process Approach

## **Overview**

This project presents a comprehensive analysis of high-frequency trading data for **INFY (Infosys)** on the National Stock Exchange (NSE). We employ a **multivariate Hawkes process** to model the arrival dynamics of **buyer-initiated** and **seller-initiated** trades.

The study combines rigorous data cleaning, aggressor side determination, and maximum likelihood estimation (MLE) to quantify the degree of endogeneity and self-excitation in the market. The results reveal significant branching ratios, indicating strong clustering of market events.

**Full Report**: See `docs/final_report.tex` (or the PDF version if available) for the detailed academic report.

---

## **Project Goals**

*   Efficiently load and parse large **NSE Tick-by-Tick Data** (Orders and Trades).
*   Reconstruct the **Limit Order Book (LOB)** context to identify trade aggressors.
*   Model trade arrival times using a **Bivariate Hawkes Process** (Buy vs. Sell).
*   Quantify market properties:
    *   **Self-Excitation**: Herding behavior (Buy triggers Buy).
    *   **Cross-Excitation**: interacting effects (Buy triggers Sell).
    *   **Branching Ratio**: Metric for market stability and endogeneity.
---

## **Repository Structure**

```
Hawkes_Modeling/
│
├── data_loader/                 # Data ingestion scripts
│   ├── NseDataLoaderOptimized.py  # Memory-efficient streaming loader for .DAT.gz files
│   ├── NseDataLoader.py           # Standard loader
│   └── process_23rd_data.py       # Utility for processing specific dates
│
├── models/                      # Core modeling logic
│   ├── aggressor_classifier.py    # Logic to determine trade aggressor (Buyer/Seller)
│   ├── hawkes_model.py            # Hawkes process model definitions
│   └── hawkes_utils.py            # Helper functions for diagnostics and plotting
│
├── preprocessing/               # Data preparation
│   ├── aggressor_calculator.py    # Calculator for aggressor side
│   └── prepare_hawkes_data.py     # Prepares event streams for model fitting
│
├── notebooks/                   # Analysis and drivers
│   ├── 01_data_load.ipynb         # Step 1: Loading raw NSE data
│   ├── 02_data_cleaning.ipynb     # Step 2: Cleaning and preprocessing
│   ├── 03_OFI_Analysis.ipynb      # Step 3: Order Flow Imbalance analysis
│   ├── 04_Visualization.ipynb     # Step 4: Visualizing results and model fits
│   ├── BivariateHawkes.py         # Custom Bivariate Hawkes implementation
│   └── MarkedBivariateHawkes.py   # Marked Hawkes implementation (Volume/OFI)
│
└── docs/                        # Project documentation and reports
    └── final_report.tex           # Academic report source
```

---

## **Methodology**

### **1. Data Processing**
*   **Source**: NSE Tick-by-Tick data (Orders and Trades).
*   **Loader**: `NseDataLoaderOptimized` reads compressed data files line-by-line to handle large volumes efficiently.
*   **Sample**: Analysis focuses on **August 13, 2019**, for symbol **INFY**.

### **2. Aggressor Determination**
To classify trades as Buyer-Initiated or Seller-Initiated, we match trade records with their corresponding orders. Since the aggressor is the party crossing the spread (arriving later):

*   Retrieve entry timestamps for the Buy Order ($t_{buy}^{entry}$) and Sell Order ($t_{sell}^{entry}$) involved in a trade.
*   **Logic**:
    *   **Buyer Initiated (+1)**: if $t_{buy}^{entry} > t_{sell}^{entry}$
    *   **Seller Initiated (-1)**: if $t_{sell}^{entry} > t_{buy}^{entry}$

*Implemented in `models/aggressor_classifier.py`.*

### **3. Multivariate Hawkes Process**
We model the intensities $\lambda_1(t)$ (Buy) and $\lambda_2(t)$ (Sell) as:

$$ \lambda_i(t) = \mu_i + \sum_{j=1}^{2} \int_{-\infty}^{t} \alpha_{ij} e^{-\beta_{ij}(t-s)} dN_j(s) $$

*   $\mu_i$: Baseline intensity (exogenous events).
*   $\alpha_{ij}$: Excitation impact from type $j$ to $i$.
*   $\beta_{ij}$: Decay rate of the impact.

---

## **Empirical Findings (INFY)**

Based on the analysis of the August 13, 2019 dataset:

*   **Self-Excitation**: Strong evidence of clustering.
    *   Buy $\to$ Buy Branching Ratio ($n_{11}$) $\approx 0.04$
    *   Sell $\to$ Sell Branching Ratio ($n_{22}$) $\approx 0.04$
*   **Stability**: The market layout is stable (spectral radius < 1), meaning order flow does not explode indefinitely.
*   **Liquidity**: The baseline intensities ($\mu$) capture fundamental liquidity demand, while branching ratios capture reactive trading.

---

## **Usage Pipeline**

### **1. Load Data**
Use the notebooks or scripts to load raw data:
```python
from data_loader.NseDataLoaderOptimized import NseDataLoaderOptimized
loader = NseDataLoaderOptimized()
orders, trades = loader.load_symbol_for_day(orders_path, trades_path, "INFY")
```

### **2. Preprocessing & Aggressor Assignment**
Run the cleaning pipeline to assign aggressor flags:
```python
# See notebooks/02_data_cleaning.ipynb for full workflow
```

### **3. Fit Hawkes Model**
Use the custom implementations in `notebooks/` to fit the model:
```python
from notebooks.BivariateHawkes import BivariateHawkes
model = BivariateHawkes(n_nodes=2)
model.fit(timestamps)
```

---

## **Dependencies**

*   **Python 3.8+**
*   `pandas`, `numpy`, `scipy`
*   `matplotlib`, `seaborn`
*   `tick` (optional, for comparison)
*   `gzip`

---

## **License**

MIT License
