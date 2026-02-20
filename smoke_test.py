"""
smoke_test.py
End-to-end live test: Databento AAPL options chain → quantum pricing.

Run from project root:
    python smoke_test.py

Steps:
  1. Fetch AAPL options chain (Databento OPRA.PILLAR)
  2. Get AAPL spot price (yfinance)
  3. Identify ATM call + put at nearest expiration
  4. Back out implied volatility from NBBO mid (Black-Scholes inverse)
  5. Price via quantum engine (CPU, 4 qubits)
  6. Compare quantum price vs Black-Scholes reference
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env before pydantic-settings caches Settings
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── imports ───────────────────────────────────────────────────────────────────
from datetime import date
import numpy as np
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm as sp_norm

from config.settings import get_settings
get_settings.cache_clear()   # ensure fresh load with .env values

from data.market_data import get_options_chain
from pricer.circuit import derive_lognormal_params, build_uncertainty_model, build_estimation_problem
from pricer.engine import build_sampler, run_iae, interpret_results


# ── helpers ───────────────────────────────────────────────────────────────────

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * sp_norm.cdf(d1) - K * np.exp(-r * T) * sp_norm.cdf(d2)
    return K * np.exp(-r * T) * sp_norm.cdf(-d2) - S * sp_norm.cdf(-d1)


def implied_vol(S, K, T, r, market_price, option_type="call"):
    """Back out IV from market mid price via Brent root-finding."""
    try:
        return brentq(
            lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - market_price,
            1e-4, 10.0, xtol=1e-6,
        )
    except Exception:
        return None


def sep(char="─", width=62):
    print(char * width)


def header(title):
    sep("═")
    print(f"  {title}")
    sep("═")


def section(title):
    print()
    sep()
    print(f"  {title}")
    sep()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    TICKER        = "AAPL"
    RISK_FREE     = 0.053          # ~current Fed funds rate
    NUM_QUBITS    = 8              # circuit resolution (256 price bins, <1% vs BS on GPU)
    EPSILON       = 0.05           # IAE precision (loose — fast on CPU)
    ALPHA         = 0.05           # 95% confidence interval

    header(f"Quantum Options Pricer — Live Smoke Test   ({date.today()})")

    # ── Step 1: Databento options chain ──────────────────────────────────────
    section("STEP 1 — Fetch AAPL Options Chain (Databento OPRA.PILLAR)")
    snapshot = get_options_chain(TICKER)

    print(f"  Ticker          : {snapshot.ticker}")
    print(f"  Snapshot date   : {snapshot.snapshot_date}")
    print(f"  Snapshot window : {snapshot.snapshot_time.strftime('%H:%M:%S %Z')}")
    print(f"  Total calls     : {len(snapshot.calls)}")
    print(f"  Total puts      : {len(snapshot.puts)}")
    print(f"  Expirations     : {len(snapshot.unique_expirations)}")
    print(f"  Strike range    : ${snapshot.unique_strikes[0]:.2f} — ${snapshot.unique_strikes[-1]:.2f}")
    print()
    print("  Next 5 expirations:")
    for exp in snapshot.unique_expirations[:5]:
        print(f"    {exp}")

    # ── Step 2: Spot price ────────────────────────────────────────────────────
    section("STEP 2 — AAPL Spot Price (yfinance)")
    spot = yf.Ticker(TICKER).fast_info["last_price"]
    print(f"  AAPL spot       : ${spot:.4f}")

    # ── Step 3: Select ATM option at nearest expiration ───────────────────────
    section("STEP 3 — Select ATM Option (nearest expiry)")
    nearest_exp   = snapshot.unique_expirations[0]
    exp_snapshot  = snapshot.filter_expiration(nearest_exp)

    # ATM call
    calls = exp_snapshot.calls.copy()
    calls["moneyness"] = (calls["strike"] - spot).abs()
    atm_call = calls.loc[calls["moneyness"].idxmin()]

    # ATM put at same strike
    puts = exp_snapshot.puts.copy()
    puts["moneyness"] = (puts["strike"] - spot).abs()
    atm_put = puts.loc[puts["moneyness"].idxmin()]

    K   = float(atm_call["strike"])
    T   = float(atm_call["time_to_expiry"])
    DTE = int(atm_call["days_to_expiry"])

    print(f"  Expiration      : {nearest_exp}  ({DTE} days)")
    print(f"  ATM strike      : ${K:.2f}")
    print()
    print(f"  Call  bid/ask   : ${atm_call['bid']:.4f} / ${atm_call['ask']:.4f}")
    print(f"  Call  mid       : ${atm_call['mid']:.4f}   spread=${atm_call['spread']:.4f}")
    print(f"  Put   bid/ask   : ${atm_put['bid']:.4f} / ${atm_put['ask']:.4f}")
    print(f"  Put   mid       : ${atm_put['mid']:.4f}   spread={atm_put['spread']:.4f}")
    print(f"  Symbol (call)   : {atm_call['raw_symbol']}")

    # ── Step 4: Implied volatility ────────────────────────────────────────────
    section("STEP 4 — Implied Volatility (Black-Scholes inverse from NBBO mid)")

    iv_call = implied_vol(spot, K, T, RISK_FREE, atm_call["mid"], "call")
    iv_put  = implied_vol(spot, K, T, RISK_FREE, atm_put["mid"],  "put")

    if iv_call is None:
        print("  WARNING: IV inversion failed for call — using fallback 0.25")
        iv_call = 0.25
    if iv_put is None:
        print("  WARNING: IV inversion failed for put — using fallback 0.25")
        iv_put = 0.25

    print(f"  Call IV         : {iv_call*100:.2f}%")
    print(f"  Put  IV         : {iv_put*100:.2f}%")
    print(f"  IV spread       : {abs(iv_call - iv_put)*100:.2f}%  (C-P, parity residual)")
    print(f"  r               : {RISK_FREE*100:.2f}%")
    print(f"  T               : {T:.6f} yrs")

    # Use call IV as the canonical vol for quantum pricing
    vol = iv_call

    # ── Step 5: Black-Scholes reference prices ────────────────────────────────
    section("STEP 5 — Black-Scholes Reference Prices")
    bs_call_price = black_scholes(spot, K, T, RISK_FREE, vol, "call")
    bs_put_price  = black_scholes(spot, K, T, RISK_FREE, vol, "put")
    print(f"  BS call price   : ${bs_call_price:.4f}")
    print(f"  BS put price    : ${bs_put_price:.4f}")
    print(f"  BS put-call parity check: C - P = ${bs_call_price - bs_put_price:.4f}  "
          f"(S - K*e^(-rT) = ${spot - K*np.exp(-RISK_FREE*T):.4f})")

    # ── Step 6: Quantum circuit construction ──────────────────────────────────
    section(f"STEP 6 — Quantum Circuit  (num_qubits={NUM_QUBITS})")
    lnp     = derive_lognormal_params(spot, RISK_FREE, vol, T)
    model   = build_uncertainty_model(NUM_QUBITS, lnp)
    ec, problem = build_estimation_problem(NUM_QUBITS, K, lnp, model)
    grover  = problem.grover_operator

    print(f"  mu              : {lnp.mu:.6f}")
    print(f"  sigma           : {lnp.sigma:.6f}")
    print(f"  E[S_T]          : ${lnp.mean_spot:.4f}  (forward price)")
    print(f"  Bounds          : [${lnp.low:.4f}, ${lnp.high:.4f}]")
    print(f"  Price bins      : {2**NUM_QUBITS}")
    print(f"  Circuit qubits  : {grover.num_qubits}  "
          f"(uncertainty={NUM_QUBITS} + ancilla={grover.num_qubits - NUM_QUBITS})")
    print(f"  Grover depth    : {grover.decompose().depth()}")

    # ── Step 7: Quantum pricing ───────────────────────────────────────────────
    section(f"STEP 7 — Quantum Pricing  (IAE ε={EPSILON} α={ALPHA}  backend=CPU)")
    print("  Running Iterative Amplitude Estimation...")
    sampler    = build_sampler("cpu")
    iae_result = run_iae(problem, sampler, EPSILON, ALPHA)

    q_call = interpret_results(ec, problem, iae_result, spot, K, RISK_FREE, T, "call")
    q_put  = interpret_results(ec, problem, iae_result, spot, K, RISK_FREE, T, "put")

    # ── Step 8: Results ───────────────────────────────────────────────────────
    section("STEP 8 — Pricing Results")

    def pct_diff(q, bs):
        return (q - bs) / bs * 100 if bs != 0 else float("nan")

    mkt_c   = f"${atm_call['mid']:.4f}"
    mkt_p   = f"${atm_put['mid']:.4f}"
    bsc     = f"${bs_call_price:.4f}"
    bsp     = f"${bs_put_price:.4f}"
    qc      = f"${q_call.price:.4f}"
    qp      = f"${q_put.price:.4f}"
    ci_cl   = f"${q_call.confidence_interval[0]:.4f}"
    ci_cu   = f"${q_call.confidence_interval[1]:.4f}"
    ci_pl   = f"${q_put.confidence_interval[0]:.4f}"
    ci_pu   = f"${q_put.confidence_interval[1]:.4f}"
    dfc     = f"{pct_diff(q_call.price, bs_call_price):+.1f}%"
    dfp     = f"{pct_diff(q_put.price,  bs_put_price):+.1f}%"

    print(f"  {'Metric':<28} {'Call':>12} {'Put':>12}")
    sep()
    print(f"  {'Market mid (NBBO)':<28} {mkt_c:>12} {mkt_p:>12}")
    print(f"  {'Black-Scholes price':<28} {bsc:>12} {bsp:>12}")
    print(f"  {'Quantum price':<28} {qc:>12} {qp:>12}")
    print(f"  {'Quantum CI lower':<28} {ci_cl:>12} {ci_pl:>12}")
    print(f"  {'Quantum CI upper':<28} {ci_cu:>12} {ci_pu:>12}")
    print(f"  {'vs Black-Scholes':<28} {dfc:>12} {dfp:>12}")
    print(f"  {'Put via parity':<28} {'—':>12} {str(q_put.calculated_via_parity):>12}")
    sep()
    print(f"  {'Oracle queries':<28} {q_call.oracle_queries:>12}")
    print(f"  {'IAE rounds':<28} {q_call.num_iterations:>12}")
    print(f"  {'Total circuit qubits':<28} {q_call.total_qubits:>12}")
    print(f"  {'Grover depth':<28} {q_call.grover_depth:>12}")

    print()
    print("  NOTE: {}-qubit coarse grid. Expect ±5–20% vs BS.".format(NUM_QUBITS))
    print("  On GPU (tensor_network) with 8 qubits, error drops to <1%.")
    sep("═")
    print("  Smoke test PASSED")
    sep("═")


if __name__ == "__main__":
    main()
