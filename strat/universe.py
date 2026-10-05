"""Investment universe: US-listed proxies with long histories (Yahoo tickers), grouped by sleeve.

Backtests run on these proxies; ``UCITS`` maps the ETFs to instruments tradable from Europe.
Known limitation: the stock list is today's large caps (survivorship bias).
"""

from __future__ import annotations

SP500 = """AAPL MSFT NVDA AMZN GOOGL META BRK-B JPM V MA UNH XOM JNJ PG HD COST ABBV KO PEP MRK
WMT BAC CVX LLY TMO MCD CSCO ACN ABT DHR LIN WFC DIS VZ TXN PM NEE UNP RTX HON
LOW IBM CAT GS AMGN SPGI BLK DE MDT T C MMM BA GE NKE UPS SBUX MO CL""".split()

NASDAQ = """AVGO TSLA NFLX ADBE AMD INTC QCOM INTU AMAT BKNG ISRG ADP GILD VRTX REGN LRCX MU ADI
KLAC SNPS CDNS MELI ORLY MAR CTAS PANW FTNT ASML MDLZ CMCSA PYPL CSX ROST MNST PCAR NXPI""".split()

UNIVERSE: dict[str, list[str]] = {
    "stocks": SP500 + NASDAQ,
    # equity ETFs + multi-asset ETFs compete in the same sleeve
    "etf": """SPY QQQ IWM EFA EEM ACWI VYM VLUE QUAL MTUM USMV CIBR SMH VNQ XLE XLV
              AOK AOM AOR AOA NTSX""".split(),
    "bonds": "IEF TLT TIP LQD HYG EMB BNDX AGG".split(),
    "commodities": "GLD SLV DBC USO CPER DBA PDBC".split(),
    "cash": ["SHY"],
}

GROUPS: dict[str, str] = {t: g for g, ts in UNIVERSE.items() for t in ts}

# Proxy -> UCITS ISIN for execution. Check each ISIN on your broker before trading.
UCITS: dict[str, str] = {
    "SPY": "IE00B5BMR087", "QQQ": "IE0032077012", "IWM": "IE00B3VWM098",
    "EEM": "IE00B4L5YC18", "ACWI": "IE00B6R52259", "VYM": "IE00B8GKDB10", "VLUE": "IE00BP3QZB59",
    "QUAL": "IE00BP3QZ601", "MTUM": "IE00BP3QZ825", "USMV": "IE00B8FHGS14", "SMH": "IE00BMC38736",
    "IEF": "IE00B1FZS798", "TLT": "IE00BSKRJZ44", "TIP": "IE00B1FZSC47", "LQD": "IE0032895942",
    "HYG": "IE00B4PY7Y77", "EMB": "IE00B2NPKV68", "AGG": "IE00B3F81409", "SHY": "IE00B14X4S71",
    "GLD": "IE00B4ND3602", "SLV": "IE00B4NCWG09", "DBC": "IE00BDFL4P12",
}


def all_tickers() -> list[str]:
    return list(GROUPS)


NAMES: dict[str, str] = {
    # --- stocks
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon", "GOOGL": "Alphabet (Class A)",
    "META": "Meta Platforms", "BRK-B": "Berkshire Hathaway (Class B)", "JPM": "JPMorgan Chase", "V": "Visa",
    "MA": "Mastercard", "UNH": "UnitedHealth Group", "XOM": "Exxon Mobil", "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble", "HD": "Home Depot", "COST": "Costco Wholesale", "ABBV": "AbbVie",
    "KO": "Coca-Cola", "PEP": "PepsiCo", "MRK": "Merck & Co", "WMT": "Walmart", "BAC": "Bank of America",
    "CVX": "Chevron", "LLY": "Eli Lilly", "TMO": "Thermo Fisher Scientific", "MCD": "McDonald's",
    "CSCO": "Cisco Systems", "ACN": "Accenture", "ABT": "Abbott Laboratories", "DHR": "Danaher",
    "LIN": "Linde", "WFC": "Wells Fargo", "DIS": "Walt Disney", "VZ": "Verizon", "TXN": "Texas Instruments",
    "PM": "Philip Morris International", "NEE": "NextEra Energy", "UNP": "Union Pacific", "RTX": "RTX (Raytheon)",
    "HON": "Honeywell", "LOW": "Lowe's", "IBM": "IBM", "CAT": "Caterpillar", "GS": "Goldman Sachs",
    "AMGN": "Amgen", "SPGI": "S&P Global", "BLK": "BlackRock", "DE": "Deere & Co", "MDT": "Medtronic",
    "T": "AT&T", "C": "Citigroup", "MMM": "3M", "BA": "Boeing", "GE": "GE Aerospace", "NKE": "Nike",
    "UPS": "United Parcel Service", "SBUX": "Starbucks", "MO": "Altria", "CL": "Colgate-Palmolive",
    "AVGO": "Broadcom", "TSLA": "Tesla", "NFLX": "Netflix", "ADBE": "Adobe", "AMD": "Advanced Micro Devices",
    "INTC": "Intel", "QCOM": "Qualcomm", "INTU": "Intuit", "AMAT": "Applied Materials",
    "BKNG": "Booking Holdings", "ISRG": "Intuitive Surgical", "ADP": "Automatic Data Processing",
    "GILD": "Gilead Sciences", "VRTX": "Vertex Pharmaceuticals", "REGN": "Regeneron", "LRCX": "Lam Research",
    "MU": "Micron Technology", "ADI": "Analog Devices", "KLAC": "KLA", "SNPS": "Synopsys",
    "CDNS": "Cadence Design Systems", "MELI": "MercadoLibre", "ORLY": "O'Reilly Automotive",
    "MAR": "Marriott International", "CTAS": "Cintas", "PANW": "Palo Alto Networks", "FTNT": "Fortinet",
    "ASML": "ASML Holding (ADR)", "MDLZ": "Mondelez International", "CMCSA": "Comcast", "PYPL": "PayPal",
    "CSX": "CSX", "ROST": "Ross Stores", "MNST": "Monster Beverage", "PCAR": "Paccar",
    "NXPI": "NXP Semiconductors",
    # --- equity and multi-asset ETFs
    "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ (Nasdaq 100)", "IWM": "iShares Russell 2000 ETF",
    "EFA": "iShares MSCI EAFE ETF", "EEM": "iShares MSCI Emerging Markets ETF", "ACWI": "iShares MSCI ACWI ETF",
    "VYM": "Vanguard High Dividend Yield ETF", "VLUE": "iShares MSCI USA Value Factor ETF",
    "QUAL": "iShares MSCI USA Quality Factor ETF", "MTUM": "iShares MSCI USA Momentum Factor ETF",
    "USMV": "iShares MSCI USA Min Vol Factor ETF", "CIBR": "First Trust Nasdaq Cybersecurity ETF",
    "SMH": "VanEck Semiconductor ETF", "VNQ": "Vanguard Real Estate ETF",
    "XLE": "Energy Select Sector SPDR", "XLV": "Health Care Select Sector SPDR",
    "AOK": "iShares Core Conservative Allocation ETF", "AOM": "iShares Core Moderate Allocation ETF",
    "AOR": "iShares Core Growth Allocation ETF", "AOA": "iShares Core Aggressive Allocation ETF",
    "NTSX": "WisdomTree U.S. Efficient Core (90/60)",
    # --- bonds
    "IEF": "iShares 7-10 Year Treasury Bond ETF", "TLT": "iShares 20+ Year Treasury Bond ETF",
    "TIP": "iShares TIPS Bond ETF", "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF", "EMB": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
    "BNDX": "Vanguard Total International Bond ETF", "AGG": "iShares Core U.S. Aggregate Bond ETF",
    # --- commodities
    "GLD": "SPDR Gold Shares", "SLV": "iShares Silver Trust", "DBC": "Invesco DB Commodity Index Tracking Fund",
    "USO": "United States Oil Fund (WTI crude)", "CPER": "United States Copper Index Fund",
    "DBA": "Invesco DB Agriculture Fund", "PDBC": "Invesco Optimum Yield Diversified Commodity (No K-1)",
    # --- cash
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
}
