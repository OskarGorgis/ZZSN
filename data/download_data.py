import yfinance as yf
import pandas as pd
import os

# ── Parametry ──────────────────────────────────────────────
START = "2021-01-01"
END   = "2026-01-31"
OUT   = "data/raw"
os.makedirs(OUT, exist_ok=True)

# ── Zbiór 1: NASDAQ-100 (aktualny skład) ───────────────────
NASDAQ100 = [
    "NVDA","AAPL","MSFT","AMZN","GOOGL","GOOG","AVGO","TSLA","META","MU",
    "WMT","AMD","ASML","INTC","CSCO","COST","LRCX","ARM","PLTR","NFLX",
    "AMAT","TXN","QCOM","KLAC","SNDK","LIN","PANW","APP","TMUS","ADI",
    "STX","PEP","CRWD","WDC","AMGN","MRVL","GILD","SHOP","HON","ISRG",
    "BKNG","PDD","VRTX","SBUX","ADBE","CEG","CDNS","FTNT","MAR","SNPS",
    "INTU","CMCSA","ADP","DDOG","MNST","MELI","CSX","NXPI","ABNB","MDLZ",
    "MPWR","ROST","ORLY","DASH","AEP","CTAS","WBD","LITE","REGN","BKR",
    "PCAR","MSTR","FANG","MCHP","FAST","EA","XEL","ADSK","FER","ODFL",
    "EXC","IDXX","TTWO","KDP","ALNY","CCEP","PYPL","TRI","AXON","WDAY",
    "PAYX","ROP","CPRT","KHC","DXCM","GEHC","CTSH","INSM","VRSK","ZS","CHTR"
]

# ── Zbiór 2: WIG20 ─────────────────────────────────────────
WIG20 = [
    "MBK.WA",   # mBank
    "BDX.WA",   # Budimex
    "SPL.WA",   # Erste Bank (ErsteGroup PL)
    "MDV.WA",   # Modivo
    "KTY.WA",   # Kety
    "KGH.WA",   # KGHM
    "LPP.WA",   # LPP
    "CDR.WA",   # CD Projekt
    "PEO.WA",   # Pekao
    "PKN.WA",   # PKN Orlen
    "PKO.WA",   # PKO BP
    "PGE.WA",   # PGE
    "PZU.WA",   # PZU
    "TPE.WA",   # Tauron
    "KRU.WA",   # Kruk
    "ALR.WA",   # Alior Bank
    "DNP.WA",   # Dino Polska
    "ALE.WA",   # Allegro
    "PCO.WA",   # Pepco
    "ZAB.WA",   # Żabka
]

# ── Zbiór 3: mWIG40 ────────────────────────────────────────
MWIG40 = [
    "ABE.WA",   # AB
    "EAT.WA",   # AmRest
    "ACP.WA",   # Asseco Poland
    "DOM.WA",   # Dom Development
    "EUR.WA",   # Eurocash
    "BHW.WA",   # Bank Handlowy
    "ING.WA",   # ING Bank Śląski
    "CAR.WA",   # Inter Cars
    "DVL.WA",   # Develia
    "LBW.WA",   # Lubawa
    "MIL.WA",   # Millennium
    "PEP.WA",   # Pepees (PEP)
    "PXM.WA",   # Polimex-Mostostal
    "NEU.WA",   # Neuca
    "OPL.WA",   # Orange Polska
    "RBW.WA",   # Rainbow Tours
    "CPS.WA",   # Cyfrowy Polsat
    "ASB.WA",   # ASBIS
    "AGT.WA",   # Grupa Azoty
    "ENA.WA",   # Enea
    "MRB.WA",   # Mirbud
    "ASE.WA",   # Asseco SEE
    "MBR.WA",   # Mobruk
    "GPW.WA",   # GPW
    "BFT.WA",   # Benefit Systems
    "JSW.WA",   # JSW
    "SNT.WA",   # Synektik
    "VOX.WA",   # Voxel
    "NWG.WA",   # Newag
    "TXT.WA",   # Text (LiveChat)
    "WPL.WA",   # Wirtualna Polska
    "BNP.WA",   # BNP Paribas Bank Polska
    "XTB.WA",   # XTB
    "APT.WA",   # Auto Partner
    "CBF.WA",   # Cyberfolks
    "TSG.WA",   # Ten Square Games
    "VRC.WA",   # Vercom
    "CRI.WA",   # Creotech
    "GPP.WA",   # Grupa Pracuj
    "DIG.WA",   # Diagnostyka
]

WIG60 = WIG20 + MWIG40

# ── Zbiór 4: NASDAQ-100 + aktywa alternatywne ─────────────
ALTERNATIVE = [
    "GLD",    # ETF złoto
    "SLV",    # ETF srebro
    "TLT",    # ETF obligacje 20Y USA
    "SHY",    # ETF obligacje 1-3Y USA
]

NASDAQ100_EXT = NASDAQ100 + ALTERNATIVE


# ── Funkcja pobierająca ────────────────────────────────────
def download(tickers, name):
    print(f"\nPobieram {name} ({len(tickers)} tickerów)...")
    df = yf.download(
        tickers,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True
    )
    path = f"{OUT}/{name}.parquet"
    df.to_parquet(path)
    print(f"Zapisano: {path}  |  shape: {df.shape}")
    return df


if __name__ == "__main__":
    download(NASDAQ100,     "nasdaq100")
    download(WIG60,         "wig60")
    download(NASDAQ100_EXT, "nasdaq100_extended")
    print("\nGotowe!")