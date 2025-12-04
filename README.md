# **Hawkes Modeling of Order-Flow Imbalance (OFI) — NSE Tick Data**

This project analyzes **high-frequency trading behavior** in NSE equities using **bivariate Hawkes processes**.
We model **buyer-initiated vs seller-initiated trades**, quantify **clustering** and **endogeneity**, and relate these patterns to **market microstructure concepts** such as liquidity, execution pressure, and short-horizon order-flow imbalance (OFI).

---

## **Project Goals**

* Efficiently load large **CASH_Orders_*.DAT.gz** and **CASH_Trades_*.DAT.gz** files
* Extract **one symbol at a time** using a **streaming, memory-safe loader**
* Identify the **aggressor side** for each trade (buyer vs seller)
* Build a **clean event stream** of buy/sell timestamps
* Fit a **2D Hawkes process** to quantify:

  * Buy→Buy and Sell→Sell clustering (herding)
  * Buy↔Sell cross-effects (contrarian reactions)
  * Endogeneity of order flow
* Generate **OFI-based pressure signals** using Hawkes intensities

---

## **Structure**

```
hawkes_project/
│
├── data_loader/
│   ├── NseDataLoaderOptimized.py   # Streaming loader for full-universe files
│
├── preprocessing/
│   ├── build_event_stream.py       # Aggressor labeling, event construction
│
├── models/
│   ├── hawkes_fit.py               # Fit Hawkes model
│   ├── hawkes_utils.py             # Diagnostics, plots
│
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_hawkes_estimation.ipynb
│   ├── 03_ofi_analysis.ipynb
│
└── README.md
```

---

## **Data Requirements**

NSE CM tick-by-tick files:

* `CASH_Orders_YYYYMMDD.DAT.gz`
* `CASH_Trades_YYYYMMDD.DAT.gz`

These contain full order-book events and trades across all securities.
We stream them line-by-line and extract only the target symbol (e.g., **INFY**).

---

## **Pipeline**

### 1. Load symbol for a day

```python
loader = NseDataLoaderOptimized()

orders_df, trades_df = loader.load_symbol_for_day(
    order_filepath="CASH_Orders_19082019.DAT.gz",
    trade_filepath="CASH_Trades_19082019.DAT.gz",
    symbol="INFY"
)
```

### 2. Build event stream

```python
events = build_event_stream(orders_df, trades_df)
```

### 3. Fit Hawkes model

```python
hawkes_results = fit_hawkes(events)
```

### 4. Inspect clustering / OFI pressure

```python
plot_intensity(hawkes_results)
print_parameters(hawkes_results)
```

---

## **Relevance**

* How buy/sell trades cluster in time
* Whether buy actions trigger more buying (herding)
* Whether sells trigger buys (mean-reversion)
* How much of order flow is **endogenous** vs **information-driven**
* How Hawkes intensities can produce **short-horizon OFI indicators**

---

## **Dependencies**

* Python 3.9+
* pandas, numpy
* tick (Hawkes modeling)
* matplotlib / seaborn
* gzip

---

## **Notes**

* Works best on **liquid symbols** (INFY, RELIANCE, TCS, etc.)
* Recommended: **10–20 days** of tick data per symbol

---

## **License**

MIT License

---
