"""Synthetic SEC filing generator, U.S. regional bank stress scenario 2020–2024.

Scenario: Eight U.S. banks across three balance-sheet-health tiers navigate COVID
loan losses, the 2022–2023 rate hiking cycle, and two bank failures (SIVB, FRC).

Latent structure: each ticker has a fixed `balance_sheet_health` score [0,1] drawn
from tier-specific ranges. This drives:
  - MD&A sentiment: positive baseline modulated by health + regime + NIM boost
  - Risk-factors sentiment: negative baseline modulated by health + regime tightening
  - In the hiking regime, low-health banks get a NIM boost to MD&A while their risk
    language deteriorates disproportionately, widening the section gap before failure.
  - SIVB and FRC receive additional terminal-stress adjustments in Q3–Q4 2022 and
    stop filing after 2022Q4 (failed in 2023Q1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ticker metadata
# ---------------------------------------------------------------------------

_TICKER_META: dict[str, dict] = {
    "JPM":  {"tier": "gsib",           "health": 0.88, "failed": False, "failure_period": None},
    "BAC":  {"tier": "gsib",           "health": 0.82, "failed": False, "failure_period": None},
    "USB":  {"tier": "super_regional", "health": 0.68, "failed": False, "failure_period": None},
    "RF":   {"tier": "super_regional", "health": 0.63, "failed": False, "failure_period": None},
    "KEY":  {"tier": "stressed",       "health": 0.52, "failed": False, "failure_period": None},
    "ZION": {"tier": "stressed",       "health": 0.48, "failed": False, "failure_period": None},
    "FRC":  {"tier": "stressed",       "health": 0.31, "failed": True,
             "failure_period": pd.Period("2023Q1")},
    "SIVB": {"tier": "stressed",       "health": 0.25, "failed": True,
             "failure_period": pd.Period("2023Q1")},
}

_TIER_HEALTH_RANGES = {
    "gsib":           (0.78, 0.95),
    "super_regional": (0.55, 0.72),
    "stressed":       (0.22, 0.55),
}


# ---------------------------------------------------------------------------
# Text templates by (section × rate-regime)
# ---------------------------------------------------------------------------

_MDA_PRE_COVID = [
    "Net interest income grew {pct}% year-over-year to ${amount} million, reflecting strong loan "
    "growth across commercial and industrial segments.",
    "We delivered return on average tangible common equity of {pct}%, supported by disciplined "
    "expense management and improving credit quality.",
    "Our loan portfolio expanded {pct}% driven by robust demand from middle-market and corporate "
    "clients investing in capacity expansion.",
    "Fee income rose {pct}% year-over-year, with wealth management and investment banking both "
    "contributing to diversified revenue growth.",
    "Non-interest expense declined {pct}% as branch optimization and technology efficiency "
    "initiatives yielded meaningful cost savings.",
    "Deposits grew ${amount} million, reflecting continued deepening of client relationships "
    "and strong market share in our core geographies.",
    "Our CET1 capital ratio stood at {pct}%, comfortably above regulatory minimums and "
    "supporting our capital return programs.",
    "Credit quality remained excellent with net charge-offs at ${amount} million, representing "
    "{bps} basis points of average loans.",
    "We returned ${amount} million to shareholders through dividends and share repurchases, "
    "reflecting confidence in the strength of our balance sheet.",
]

_MDA_COVID_STRESS = [
    "We recorded a provision for credit losses of ${amount} million, significantly above prior "
    "quarter levels, reflecting severe deterioration in macroeconomic conditions due to COVID-19.",
    "Despite a challenging operating environment, net interest income held at ${amount} million, "
    "supported by Paycheck Protection Program originations of ${amount} million.",
    "Our net interest margin compressed {bps} basis points as Federal Reserve emergency rate "
    "cuts reduced yields on variable-rate assets faster than deposit costs declined.",
    "Loan balances grew ${amount} million as commercial clients drew on revolving credit "
    "facilities to preserve liquidity during the pandemic.",
    "Fee revenue declined {pct}% as economic activity slowed, reducing transaction volumes "
    "and wealth management revenue tied to equity market levels.",
    "We suspended share repurchases to preserve capital and support lending to clients "
    "through the economic disruption.",
    "Our allowance for credit losses increased ${amount} million to reflect the heightened "
    "risk of default across commercial real estate, hospitality, and retail exposures.",
    "PPP loan originations totaled ${amount} million, enabling support for approximately "
    "{amount} thousand small businesses and their employees.",
    "We maintained strong liquidity with ${amount} billion in available sources, including "
    "Federal Home Loan Bank capacity and Federal Reserve facilities.",
]

_MDA_RECOVERY = [
    "Revenue grew {pct}% year-over-year to ${amount} million, reflecting strong loan demand, "
    "improving credit quality, and reserve releases as the recovery exceeded our expectations.",
    "We released ${amount} million from our allowance for credit losses, reflecting improved "
    "macroeconomic forecasts and better-than-anticipated borrower performance.",
    "Net interest margin remained under pressure at {pct}% but we are well-positioned to "
    "benefit from anticipated Federal Reserve rate increases.",
    "Fee income grew {pct}% as investment banking activity surged, mortgage origination "
    "reached record levels, and wealth management benefited from equity market appreciation.",
    "Loan growth accelerated to {pct}% annualized as business investment and consumer "
    "spending recovered strongly from pandemic lows.",
    "We resumed share repurchases in the quarter, buying back ${amount} million of common "
    "stock as the capital outlook improved and regulatory constraints were lifted.",
    "Credit quality improved materially, with non-performing assets declining {pct}% and "
    "net charge-offs falling to {bps} basis points of average loans.",
    "Digital account openings grew {pct}% and digital transaction penetration reached "
    "{pct}% of total transactions, reflecting the returns on our technology investments.",
    "Strong deposit growth of ${amount} million continued as consumers and businesses "
    "maintained elevated savings accumulated during the pandemic.",
]

_MDA_HIKING = [
    "Net interest income grew {pct}% year-over-year to ${amount} million, driven by Federal "
    "Reserve rate increases that expanded our asset-sensitive net interest margin.",
    "Our net interest margin expanded {bps} basis points to {pct}%, as loan repricing "
    "outpaced the rise in deposit costs and our asset-sensitive positioning benefited.",
    "Fee income declined {pct}% as the rate environment reduced mortgage origination volumes "
    "and investment banking activity slowed in the more uncertain macro environment.",
    "Despite Federal Reserve rate increases, deposit betas have remained below {pct}%, "
    "allowing us to capture significant benefit from the rising rate environment.",
    "Credit quality remains strong with net charge-offs of {bps} basis points, though we "
    "are monitoring commercial real estate office and multifamily exposures carefully.",
    "We generated return on tangible common equity of {pct}%, well above our cost of capital, "
    "and returned ${amount} million to shareholders through repurchases and dividends.",
    "Our held-to-maturity securities portfolio of ${amount} million carries estimated "
    "unrealized losses reflecting the rise in rates, though positions will mature at par.",
    "We increased our provision for credit losses by ${amount} million, reflecting a more "
    "cautious macroeconomic outlook as higher rates begin to weigh on activity.",
    "Loan growth moderated to {pct}% as higher borrowing costs reduced demand in "
    "interest-rate-sensitive sectors including residential mortgage and commercial real estate.",
    "We are monitoring deposit flows carefully as some clients migrate balances to "
    "higher-yielding money market alternatives, though core deposit retention has been strong.",
]

_MDA_POST_HIKE = [
    "We delivered net interest income of ${amount} million, stabilizing following the "
    "rapid rate cycle as deposit costs and asset yields have reached equilibrium.",
    "Credit quality remained resilient with non-performing assets at ${amount} million, "
    "as our underwriting standards limited losses through the cycle.",
    "Our CET1 capital ratio strengthened to {pct}%, reflecting earnings retention and "
    "declining unrealized losses as rates stabilized near current levels.",
    "We resumed loan growth at {pct}% annualized as the rate environment stabilized "
    "and borrower confidence improved following the banking sector turbulence of early 2023.",
    "Fee income recovered {pct}% as capital markets activity improved and refinancing "
    "activity increased in anticipation of potential rate reductions.",
    "Deposit pricing has stabilized and we expect modest NIM improvement as higher-rate "
    "promotional deposits mature and reprice to current market levels.",
    "Investment securities unrealized losses declined ${amount} million as rates stabilized, "
    "partially rebuilding accumulated other comprehensive income.",
    "Technology investment continues at ${amount} million annually as we modernize our core "
    "banking infrastructure and enhance digital and AI-assisted capabilities.",
    "The banking system has stabilized following the events of 2023 and we are seeing "
    "normalized customer behavior across our deposit and lending franchises.",
]

_RISK_PRE_COVID = [
    "We face competition from large national banks, regional banks, credit unions, and "
    "non-bank financial technology companies that could adversely affect deposits and margins.",
    "Changes in interest rates, particularly an inverted yield curve, could adversely affect "
    "our net interest income, which remains our largest single revenue component.",
    "Credit losses could increase if economic conditions deteriorate, borrowers' financial "
    "conditions worsen, or collateral values decline beyond our modeled scenarios.",
    "We are subject to extensive regulation by the Federal Reserve, OCC, FDIC, and state "
    "regulators; changes in regulatory requirements could increase our cost of compliance.",
    "Cybersecurity incidents could result in significant financial losses, reputational harm, "
    "and regulatory penalties, despite our ongoing investment in security infrastructure.",
    "Concentration in commercial real estate lending of ${amount} million exposes us to "
    "sector-specific risks including vacancy rate changes and cap rate compression.",
    "Our risk models may underestimate losses during periods of economic stress because "
    "they are calibrated to historical data that may not reflect future conditions.",
    "Failure to attract and retain qualified personnel, particularly in technology roles, "
    "could adversely affect the pace and quality of our digital transformation.",
    "Litigation and regulatory investigations could result in significant financial penalties "
    "and reputational damage beyond what our current legal reserves anticipate.",
]

_RISK_COVID_STRESS = [
    "The COVID-19 pandemic has created unprecedented macroeconomic disruption; our credit "
    "loss provision of ${amount} million may prove insufficient if the recession deepens.",
    "Significant uncertainty remains about the duration and severity of the COVID-19 pandemic, "
    "making it difficult to estimate the full impact on our credit quality and earnings.",
    "Our commercial real estate portfolio, including hotel and retail exposures totaling "
    "${amount} million, is particularly vulnerable to pandemic-related economic disruption.",
    "Government stimulus programs may mask underlying credit deterioration; when programs expire, "
    "we may experience a sudden increase in delinquencies and defaults.",
    "Elevated unemployment of {pct}% increases the risk of consumer loan delinquencies, "
    "particularly in our auto, home equity, and personal lending portfolios.",
    "Our capital ratios could decline if credit losses exceed our allowance, potentially "
    "requiring us to reduce dividends or issue additional common equity at dilutive prices.",
    "Liquidity risk is elevated; we are uncertain whether capital markets will remain open "
    "and accessible if credit conditions deteriorate further.",
    "Stress testing under severe adverse scenarios indicates potential losses of ${amount} million "
    "over nine quarters, and actual losses could differ materially from modeled outcomes.",
    "COVID-related fraud and cyber threats have increased; we have heightened monitoring "
    "but cannot guarantee detection of all attempts against our systems and clients.",
]

_RISK_RECOVERY = [
    "While economic conditions have improved, the pace of recovery remains uncertain; a "
    "resurgence of COVID-19 variants could reverse positive trends in credit quality.",
    "The Federal Reserve is expected to begin reducing monetary accommodation; the pace "
    "and magnitude of rate increases could affect asset values and borrower debt capacity.",
    "Elevated asset prices across real estate, equity markets, and credit could increase "
    "the risk of a sharp correction that would impair collateral values and net worth.",
    "Labor market tightness is increasing wage costs and creating retention challenges "
    "that could adversely affect our expense base and operational capacity.",
    "Inflation risks are rising; if inflation proves persistent, the Federal Reserve may "
    "tighten monetary policy more aggressively than currently anticipated by markets.",
    "Regulatory capital requirements under Basel III finalization could materially increase "
    "our required capital, limiting capacity for loan growth and capital returns.",
    "Interest rate risk: if the yield curve steepens faster than modeled, our deposit "
    "costs could rise more rapidly than asset repricing, compressing our net interest margin.",
    "Commercial real estate faces increasing environmental sustainability requirements "
    "that could affect property values and the refinancing capacity of our borrowers.",
    "Any failure of our control environment or third-party vendor management could result "
    "in operational losses or regulatory censure from our banking supervisors.",
]

_RISK_HIKING = [
    "Rising interest rates have generated unrealized losses of ${amount} million in our "
    "available-for-sale and held-to-maturity securities portfolios.",
    "If depositors respond to higher market rates by moving balances to competitors or "
    "money market funds, we may need to raise deposit rates or accept balance outflows.",
    "Our held-to-maturity portfolio of ${amount} million carries unrealized losses; if we "
    "were required to liquidate these securities, we would crystallize significant losses.",
    "Deposit beta assumptions in our interest rate risk models may underestimate actual "
    "deposit cost increases, which could reduce projected net interest income.",
    "Commercial real estate, particularly office properties, faces structural headwinds "
    "from remote work adoption; our office loan exposure of ${amount} million carries risk.",
    "The Federal Reserve's quantitative tightening is reducing system liquidity and could "
    "increase competition for deposits, putting upward pressure on our funding costs.",
    "Concentration in deposits from technology and venture-backed companies creates "
    "correlated outflow risk if the technology funding environment deteriorates.",
    "Mark-to-market losses on our securities portfolio, while excluded from regulatory "
    "capital under current rules, represent a vulnerability if we need emergency liquidity.",
    "If macroeconomic conditions deteriorate due to Federal Reserve overtightening, credit "
    "quality in our commercial real estate, consumer, and leveraged lending portfolios "
    "could deteriorate rapidly and require materially higher provisioning.",
    "Wholesale funding costs have increased significantly; reliance on brokered deposits "
    "or FHLB advances as a substitute for core deposit outflows would compress our margins "
    "and could signal balance sheet stress to counterparties and depositors.",
    "Our interest rate risk models assume deposit betas below {pct}%; a significant "
    "deviation from this assumption would reduce net interest income and impair guidance.",
]

_RISK_POST_HIKE = [
    "While the acute stress from the Federal Reserve rate cycle has moderated, our securities "
    "portfolio still carries unrealized losses that would crystallize in a forced-sale scenario.",
    "Commercial real estate credit risk remains elevated as properties financed at lower cap "
    "rates must now refinance at higher interest rates, creating potential for borrower default.",
    "The banking sector disruptions of 2023 may have permanently altered depositor sensitivity "
    "to bank credit risk, increasing the probability of rapid outflows in future stress events.",
    "We continue to face regulatory scrutiny following the 2023 banking events, and enhanced "
    "liquidity and capital requirements could increase our cost of funds.",
    "Our net interest margin may compress if the Federal Reserve reduces rates more rapidly "
    "than modeled, as deposit rates tend to decline more slowly than loan and investment yields.",
    "Regulatory requirements for long-term debt issuance and resolution planning will increase "
    "our cost of funds and administrative burden going forward.",
    "Credit quality deterioration in office commercial real estate may not yet be fully "
    "reflected in market pricing; our exposure of ${amount} million may require additional reserves.",
    "Competition from large technology companies and fintechs in payment and deposit products "
    "continues to intensify, threatening our fee income and core deposit relationships.",
    "A rapid decline in interest rates, while beneficial for credit quality, would compress "
    "net interest margins faster than we can reprice our deposit base downward.",
]

# Terminal stress, used for SIVB and FRC in their final 1–2 quarters
_RISK_TERMINAL = [
    "Recent events in the banking sector have heightened depositor sensitivity to bank-specific "
    "risk; we have experienced elevated balance inquiries and outflows from certain client segments.",
    "Our deposit base is concentrated in the technology and venture capital ecosystem; a "
    "contraction in venture funding could trigger correlated deposit outflows across clients.",
    "The unrealized loss position in our held-to-maturity securities portfolio represents a "
    "material contingent liability if we are required to liquidate securities to meet outflows.",
    "We may need to raise additional capital or execute balance sheet repositioning actions "
    "that could be dilutive to existing shareholders or constrain future earnings capacity.",
    "Accelerated deposit outflows beyond our modeled stress scenarios could require access to "
    "costly wholesale funding, FHLB advances, or Federal Reserve emergency facilities.",
    "Market participants may interpret balance sheet repositioning actions, including potential "
    "securities sales, as a signal of financial distress, creating a self-fulfilling dynamic.",
    "Our ability to retain uninsured deposits above the FDIC limit depends on maintaining "
    "confidence in our financial strength; adverse perception could trigger rapid outflows.",
    "We cannot guarantee that available funding sources will be sufficient to replace deposit "
    "outflows if they materialize at the pace and magnitude of our more severe stress scenarios.",
]

_SECTION_TEMPLATE_MAP: dict[tuple[str, str], list[str]] = {
    ("mda", "pre_covid"):           _MDA_PRE_COVID,
    ("mda", "covid_stress"):        _MDA_COVID_STRESS,
    ("mda", "recovery"):            _MDA_RECOVERY,
    ("mda", "hiking"):              _MDA_HIKING,
    ("mda", "post_hike"):           _MDA_POST_HIKE,
    ("risk_factors", "pre_covid"):  _RISK_PRE_COVID,
    ("risk_factors", "covid_stress"): _RISK_COVID_STRESS,
    ("risk_factors", "recovery"):   _RISK_RECOVERY,
    ("risk_factors", "hiking"):     _RISK_HIKING,
    ("risk_factors", "post_hike"):  _RISK_POST_HIKE,
}


def _get_regime(period: pd.Period) -> str:
    if period <= pd.Period("2020Q1"):
        return "pre_covid"
    if period <= pd.Period("2020Q4"):
        return "covid_stress"
    if period <= pd.Period("2021Q4"):
        return "recovery"
    if period <= pd.Period("2023Q2"):
        return "hiking"
    return "post_hike"


def _compute_true_sentiment(
    section: str,
    regime: str,
    health: float,
    period: pd.Period,
    failed: bool,
    failure_period: pd.Period | None,
    rng: np.random.Generator,
) -> float:
    """
    Planted ground-truth sentiment score for a (ticker, section, quarter) cell.

    Key design intention:
    - health in [0,1] drives baseline positivity linearly
    - "mda" is structurally more positive than "risk_factors"
    - In the hiking regime, low-health banks receive a NIM boost in MD&A that
      partially offsets their health penalty, while risk-factor language tightens
      more severely for low-health banks, widening the section gap (the tension)
    - SIVB and FRC receive terminal-stress adjustments in Q3–Q4 2022
    """
    health_component = 0.50 * health - 0.25   # maps [0,1] → [-0.25, +0.25]
    section_base = 0.15 if section == "mda" else -0.10

    if section == "mda":
        regime_effect = {
            "pre_covid":    0.10,
            "covid_stress": -0.20,
            "recovery":     0.18,
            "hiking":       0.12 + (1.0 - health) * 0.12,  # NIM boost, larger for low-health
            "post_hike":    0.06,
        }[regime]
    else:  # risk_factors
        regime_effect = {
            "pre_covid":    -0.05,
            "covid_stress": -0.28,
            "recovery":     0.04,
            "hiking":       -0.22 - (1.0 - health) * 0.18,  # deeper tightening for low-health
            "post_hike":    -0.10,
        }[regime]

    # Terminal stress for SIVB/FRC: final 2 quarters before failure
    terminal = 0.0
    if failed and failure_period is not None:
        quarters_to_failure = (failure_period - period).n
        if quarters_to_failure == 1:
            terminal = -0.30
        elif quarters_to_failure == 2:
            terminal = -0.15

    raw = health_component + section_base + regime_effect + terminal
    noise = float(rng.normal(0, 0.055))
    return float(np.clip(raw + noise, -0.92, 0.92))


def _build_text(section: str, regime: str, is_terminal: bool, rng: np.random.Generator) -> str:
    """Sample and fill text templates for a given (section, regime, terminal) combination."""
    templates = list(_SECTION_TEMPLATE_MAP.get((section, regime), _RISK_PRE_COVID))

    # For terminal quarters, prepend 2 terminal stress sentences to risk_factors
    if is_terminal and section == "risk_factors":
        n_terminal = int(rng.integers(2, 4))
        terminal_pool = list(_RISK_TERMINAL)
        terminal_sentences = [
            _fill_template(t, rng)
            for t in rng.choice(terminal_pool, size=min(n_terminal, len(terminal_pool)), replace=False)
        ]
        n_reg = int(rng.integers(3, 6))
        regular = [
            _fill_template(t, rng)
            for t in rng.choice(templates, size=min(n_reg, len(templates)), replace=False)
        ]
        return " ".join(terminal_sentences + regular)

    n_sentences = int(rng.integers(4, 8))
    chosen = rng.choice(templates, size=min(n_sentences, len(templates)), replace=False)
    return " ".join(_fill_template(t, rng) for t in chosen)


def _fill_template(template: str, rng: np.random.Generator) -> str:
    pct = int(rng.integers(3, 45))
    amount = int(rng.integers(50, 8000))
    bps = int(rng.integers(8, 180))
    return template.format(pct=pct, amount=amount, bps=bps)


def _get_ticker_meta(ticker: str, rng: np.random.Generator) -> dict:
    """Return metadata for a ticker, generating defaults for unknown tickers."""
    if ticker in _TICKER_META:
        return _TICKER_META[ticker].copy()
    # Unknown ticker: assign to super_regional tier with drawn health
    lo, hi = _TIER_HEALTH_RANGES["super_regional"]
    return {
        "tier": "super_regional",
        "health": float(rng.uniform(lo, hi)),
        "failed": False,
        "failure_period": None,
    }


def generate_synthetic_corpus(
    tickers: list[str],
    start_date: str = "2020-01-01",
    end_date: str = "2024-03-31",
    quarters_per_ticker: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per (ticker, quarter, section), columns:
      ticker, filing_date, quarter, year_quarter, section, text,
      true_sentiment, tier, balance_sheet_health, rate_regime, failed_bank

    If quarters_per_ticker is provided, the function uses a simplified uniform
    generator (backwards-compatible mode for smoke tests with generic tickers).
    Otherwise it uses the scenario-based banking generator.
    """
    rng = np.random.default_rng(seed)

    if quarters_per_ticker is not None:
        return _generate_uniform(tickers, start_date, end_date, quarters_per_ticker, rng)

    return _generate_scenario(tickers, start_date, end_date, rng)


def _generate_uniform(
    tickers: list[str],
    start_date: str,
    end_date: str,
    quarters_per_ticker: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simplified uniform generator, backwards-compatible mode for smoke tests."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - start).days
    records = []

    for ticker in tickers:
        meta = _get_ticker_meta(ticker, rng)
        for q in range(quarters_per_ticker):
            offset = int(total_days / quarters_per_ticker * q)
            filing_date = start + pd.Timedelta(days=offset)
            period = pd.Period(filing_date, "Q")
            regime = _get_regime(period)

            for section in ["risk_factors", "mda"]:
                ts = _compute_true_sentiment(
                    section, regime, meta["health"], period,
                    meta["failed"], meta["failure_period"], rng,
                )
                text = _build_text(section, regime, is_terminal=False, rng=rng)
                records.append({
                    "ticker": ticker,
                    "filing_date": filing_date,
                    "quarter": str(period),
                    "year_quarter": period,
                    "section": section,
                    "text": text,
                    "true_sentiment": ts,
                    "tier": meta["tier"],
                    "balance_sheet_health": meta["health"],
                    "rate_regime": regime,
                    "failed_bank": meta["failed"],
                })

    return pd.DataFrame(records).sort_values(["ticker", "filing_date"]).reset_index(drop=True)


def _generate_scenario(
    tickers: list[str],
    start_date: str,
    end_date: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Full scenario-based generator for the regional bank stress scenario."""
    start = pd.Period(start_date, "Q")
    end = pd.Period(end_date, "Q")

    # All calendar quarters in the range
    all_periods = list(pd.period_range(start=start, end=end, freq="Q"))
    records = []

    for ticker in tickers:
        meta = _get_ticker_meta(ticker, rng)
        failure_period = meta["failure_period"]

        for period in all_periods:
            # Failed banks stop filing before their failure quarter
            if meta["failed"] and failure_period is not None and period >= failure_period:
                continue

            filing_date = pd.Timestamp(period.to_timestamp(how="end").date())
            regime = _get_regime(period)

            # Is this a terminal-stress quarter?
            quarters_to_failure = (
                (failure_period - period).n
                if (meta["failed"] and failure_period is not None)
                else 999
            )
            is_terminal = quarters_to_failure in (1, 2)

            for section in ["risk_factors", "mda"]:
                ts = _compute_true_sentiment(
                    section, regime, meta["health"], period,
                    meta["failed"], failure_period, rng,
                )
                text = _build_text(section, regime, is_terminal=is_terminal, rng=rng)
                records.append({
                    "ticker": ticker,
                    "filing_date": filing_date,
                    "quarter": str(period),
                    "year_quarter": period,
                    "section": section,
                    "text": text,
                    "true_sentiment": ts,
                    "tier": meta["tier"],
                    "balance_sheet_health": meta["health"],
                    "rate_regime": regime,
                    "failed_bank": meta["failed"],
                })

    df = pd.DataFrame(records).sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    return df
