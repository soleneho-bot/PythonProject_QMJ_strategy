import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
from scipy import stats
from hurst import compute_Hc
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# ============================================================
# 0. CONFIGURATION SUPABASE
# ============================================================
DB_URL = "postgresql://postgres.hcuzqfwmddnekcefjfky:222PROJECTSVSJ222@aws-1-eu-west-1.pooler.supabase.com:6543/postgres"

try:
    engine = create_engine(DB_URL)
    with engine.connect() as connection:
        print("✅ Successfully connected to Supabase!")
except Exception as e:
    print(f"❌ SQL Config Error: {e}")
    raise SystemExit

# ============================================================
# 1. OUTILS TICKERS & CACHE
# ============================================================
def canonical_to_yf(ticker: str) -> str:
    return ticker.replace('.', '-')

def yf_to_canonical(yf_ticker: str) -> str:
    return yf_ticker.replace('-', '.')

def _load_df_from_cache_field(field) -> pd.DataFrame:
    if not field: return pd.DataFrame()
    try:
        if isinstance(field, str): return pd.read_json(io.StringIO(field))
        else: return pd.read_json(io.StringIO(json.dumps(field)))
    except: return pd.DataFrame()

def _safe_load_info(raw_info):
    if raw_info is None: return {}
    if isinstance(raw_info, dict): return raw_info
    if isinstance(raw_info, str):
        try: return json.loads(raw_info)
        except: return {}
    return {}

# ============================================================
# 2. STRUCTURE BASE DE DONNÉES
# ============================================================
def init_db():
    with engine.connect() as conn:
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS prices (
                "Date" TIMESTAMP, "Ticker" TEXT, "Open" DOUBLE PRECISION,
                "High" DOUBLE PRECISION, "Low" DOUBLE PRECISION,
                "Close" DOUBLE PRECISION, "Volume" BIGINT,
                PRIMARY KEY ("Date", "Ticker")
            );
        '''))
        # ✅ Mise à jour de la table pour les 6 piliers
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS qmj_scores (
                "Date" DATE, "Ticker" TEXT, "Quality_Score" DOUBLE PRECISION,
                "z_Profitability" DOUBLE PRECISION, "z_Growth" DOUBLE PRECISION,
                "z_Safety" DOUBLE PRECISION, "z_Payout" DOUBLE PRECISION,
                "z_Hurst" DOUBLE PRECISION, "z_Toxic" DOUBLE PRECISION,
                "Price" DOUBLE PRECISION, PRIMARY KEY ("Date", "Ticker")
            );
        '''))
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS financial_cache (
                "Ticker" TEXT, "Ticker_YF" TEXT, "Last_Updated" TIMESTAMP,
                "Financials" JSON, "Balance_Sheet" JSON, "Cashflow" JSON, "Info" JSON,
                PRIMARY KEY ("Ticker", "Last_Updated")
            );
        '''))
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS fundamental_metrics (
                "Date" DATE, "Ticker" TEXT, "Prof_GPOA" DOUBLE PRECISION,
                "Prof_ROE" DOUBLE PRECISION, "Prof_ROA" DOUBLE PRECISION,
                "Prof_CFOA" DOUBLE PRECISION, "Prof_Accruals" DOUBLE PRECISION,
                "Grow_Earn" DOUBLE PRECISION, "Grow_Rev" DOUBLE PRECISION,
                "Safe_Beta" DOUBLE PRECISION, "Safe_IVol" DOUBLE PRECISION,
                "Safe_Lev" DOUBLE PRECISION, "Hurst" DOUBLE PRECISION,
                "Toxic_correl" DOUBLE PRECISION, "Pay_Yield" DOUBLE PRECISION,
                "Pay_Ratio" DOUBLE PRECISION, PRIMARY KEY ("Date", "Ticker")
            );
        '''))
        try:
            # Ajout des colonnes si elles manquent sur une table existante
            conn.execute(text('ALTER TABLE qmj_scores ADD COLUMN IF NOT EXISTS "z_Hurst" DOUBLE PRECISION;'))
            conn.execute(text('ALTER TABLE qmj_scores ADD COLUMN IF NOT EXISTS "z_Toxic" DOUBLE PRECISION;'))
        except: pass
        conn.commit()
    print("✅ Base de données initialisée (Modèle 6-Piliers).")

# ============================================================
# 3. RÉCUPÉRATION PRIX & FONDAMENTAUX
# ============================================================
def update_prices_smart(tickers_canonical):
    print(f"\n--- 1. MISE À JOUR PRIX ({len(tickers_canonical)} tickers) ---")
    try:
        last_date_df = pd.read_sql('SELECT MAX("Date") as last_date FROM prices', engine)
        last_date_ts = last_date_df['last_date'].iloc[0]
    except: last_date_ts = None

    if last_date_ts and pd.to_datetime(last_date_ts).date() >= datetime.now().date():
        print("-> Prix déjà à jour.")
        return

    start_dl = (pd.to_datetime(last_date_ts) - timedelta(days=2)) if last_date_ts else (datetime.now() - timedelta(days=730))
    ticker_map = {canonical_to_yf(t): t for t in tickers_canonical}
    ticker_map.update({'SPY': 'SPY', 'HYG': 'HYG'})

    try:
        new_data = yf.download(list(ticker_map.keys()), start=start_dl, interval="1d", group_by='ticker', auto_adjust=True)
        data_to_insert = []
        for yf_ticker, canonical in ticker_map.items():
            df_t = new_data[yf_ticker].copy() if len(ticker_map) > 1 else new_data.copy()
            df_t = df_t.dropna().reset_index()
            if df_t.empty: continue
            df_t['Ticker'] = canonical
            df_t = df_t.rename(columns={c: c.capitalize() for c in df_t.columns if c.lower() in ['date','open','high','low','close','volume']})
            if last_date_ts: df_t = df_t[df_t['Date'] > pd.to_datetime(last_date_ts)]
            if not df_t.empty: data_to_insert.append(df_t[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']])

        if data_to_insert:
            pd.concat(data_to_insert).drop_duplicates(subset=['Date', 'Ticker']).to_sql('prices', engine, if_exists='append', index=False, method='multi')
            print("-> Prix insérés.")
    except Exception as e: print(f"❌ Erreur prix: {e}")

def get_financial_data(ticker_canonical, ttl_days=30):
    query = text('SELECT * FROM financial_cache WHERE "Ticker" = :t ORDER BY "Last_Updated" DESC LIMIT 1')
    with engine.connect() as conn: res = conn.execute(query, {"t": ticker_canonical}).first()

    if res and (datetime.now() - pd.to_datetime(res.Last_Updated).tz_localize(None)).days < ttl_days:
        return _load_df_from_cache_field(res.Financials), _load_df_from_cache_field(res.Balance_Sheet), _load_df_from_cache_field(res.Cashflow), _safe_load_info(res.Info), True

    try:
        s = yf.Ticker(canonical_to_yf(ticker_canonical))
        f, b, c, i = s.financials, s.balance_sheet, s.cashflow, s.info
        with engine.begin() as conn:
            conn.execute(text('INSERT INTO financial_cache VALUES (:t, :ty, :d, :f, :b, :c, :i)'),
                {"t":ticker_canonical, "ty":canonical_to_yf(ticker_canonical), "d":datetime.now(), "f":f.to_json(), "b":b.to_json(), "c":c.to_json(), "i":json.dumps(i)})
        return f, b, c, i, False
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, False

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    return pd.read_html(io.StringIO(res.text))[0]['Symbol'].tolist()

# ============================================================
# 4. FONCTIONS STATISTIQUES
# ============================================================
def calc_beta(y, x):
    try: return stats.linregress(x.loc[y.index], y)[0]
    except: return np.nan

def calc_idio_vol(y, x):
    try:
        res = stats.linregress(x.loc[y.index], y)
        return (y - (res.intercept + res.slope * x.loc[y.index])).std() * np.sqrt(252)
    except: return np.nan

def calc_hurst_exponent(series):
    series = series.dropna()
    if len(series) < 60: return np.nan
    try:
        H, _, _ = compute_Hc(series, kind='price', simplified=True)
        return H
    except: return np.nan

def calc_toxic_correlation(s_ret, t_ret):
    comb = pd.concat([s_ret, t_ret], axis=1).dropna()
    return comb.iloc[:, 0].corr(comb.iloc[:, 1]) if len(comb) >= 60 else np.nan

# ============================================================
# 5. PIPELINE PRINCIPAL (LOGIQUE 6 PILIERS)
# ============================================================
def main():
    init_db()
    tickers = get_sp500_tickers()[:100]
    update_prices_smart(tickers)

    prices_df = pd.read_sql('SELECT "Date", "Ticker", "Close" FROM prices', engine).pivot(index='Date', columns='Ticker', values='Close')
    returns_df = prices_df.pct_change().dropna()
    mkt_ret = returns_df['SPY'] if 'SPY' in returns_df.columns else pd.Series()
    tox_ret = returns_df['HYG'] if 'HYG' in returns_df.columns else pd.Series()

    data = []
    print("\n--- 2. ANALYSE FONDAMENTALE ---")
    for ticker in tickers:
        if ticker in ['SPY', 'HYG']: continue
        fin, bs, cf, info, is_cached = get_financial_data(ticker)
        print("•" if is_cached else "⬇", end="", flush=True)

        try:
            curr_bs = bs.iloc[:, 0] if not bs.empty else pd.Series()
            curr_fin = fin.iloc[:, 0] if not fin.empty else pd.Series()
            assets = curr_bs.get('Total Assets', np.nan)
            debt = curr_bs.get('Total Debt', np.nan)
            ni = curr_fin.get('Net Income', np.nan)
            gp = curr_fin.get('Gross Profit', np.nan)
            ocf = info.get('operatingCashflow', np.nan)

            beta, ivol, hurst, toxic = np.nan, np.nan, np.nan, np.nan
            if ticker in returns_df.columns:
                beta = calc_beta(returns_df[ticker], mkt_ret)
                ivol = calc_idio_vol(returns_df[ticker], mkt_ret)
                toxic = calc_toxic_correlation(returns_df[ticker], tox_ret)
            if ticker in prices_df.columns:
                hurst = calc_hurst_exponent(prices_df[ticker])

            data.append({
                'Symbol': ticker, 'Price': prices_df[ticker].iloc[-1] if ticker in prices_df.columns else np.nan,
                'MarketCap': info.get('marketCap', np.nan),
                'Prof_GPOA': gp/assets if assets > 0 else np.nan,
                'Prof_ROE': info.get('returnOnEquity', np.nan),
                'Prof_ROA': info.get('returnOnAssets', np.nan),
                'Prof_CFOA': ocf/assets if assets > 0 else np.nan,
                'Prof_Accruals': (ni - ocf)/assets if assets > 0 else np.nan,
                'Grow_Earn': info.get('earningsGrowth', np.nan),
                'Grow_Rev': info.get('revenueGrowth', np.nan),
                'Safe_Beta': beta, 'Safe_IVol': ivol, 'Safe_Lev': debt/assets if assets > 0 else np.nan,
                'Safe_Hurst': hurst, 'Safe_Toxic': toxic,
                'Pay_Yield': info.get('dividendYield', 0), 'Pay_Ratio': info.get('payoutRatio', 0)
            })
        except: print("x", end="")

    df_c = pd.DataFrame(data).dropna(subset=['Price'])
    for c in df_c.select_dtypes(include=np.number).columns: df_c[c] = df_c[c].fillna(df_c[c].median())

    df_c['Size_Group'] = np.where(df_c['MarketCap'] >= df_c['MarketCap'].median(), 'Big', 'Small')
    z_rank = lambda s: (s.rank() - s.rank().mean()) / s.rank().std() if s.count() > 1 else 0

    # --- CALCUL DES 6 Z-SCORES INDÉPENDANTS ---
    df_c['z_Prof'] = (df_c.groupby('Size_Group')['Prof_GPOA'].transform(z_rank) + df_c.groupby('Size_Group')['Prof_ROE'].transform(z_rank) + df_c.groupby('Size_Group')['Prof_ROA'].transform(z_rank) + df_c.groupby('Size_Group')['Prof_CFOA'].transform(z_rank) - df_c.groupby('Size_Group')['Prof_Accruals'].transform(z_rank)) / 5
    df_c['z_Grow'] = (df_c.groupby('Size_Group')['Grow_Earn'].transform(z_rank) + df_c.groupby('Size_Group')['Grow_Rev'].transform(z_rank)) / 2

    # Safe : Uniquement Beta, IVol et Levier
    df_c['z_Safe'] = (-df_c.groupby('Size_Group')['Safe_Beta'].transform(z_rank) - df_c.groupby('Size_Group')['Safe_IVol'].transform(z_rank) - df_c.groupby('Size_Group')['Safe_Lev'].transform(z_rank)) / 3

    df_c['z_Pay'] = (df_c.groupby('Size_Group')['Pay_Yield'].transform(z_rank) + df_c.groupby('Size_Group')['Pay_Ratio'].transform(z_rank)) / 2

    # Hurst
    df_c['z_Hurst'] = df_c.groupby('Size_Group')['Safe_Hurst'].transform(z_rank)

     # Toxic
    df_c['z_Toxic'] = -df_c.groupby('Size_Group')['Safe_Toxic'].transform(z_rank)

    # SCORE FINAL SUR 6
    df_c['Quality_Score'] = (df_c['z_Prof'] + df_c['z_Grow'] + df_c['z_Safe'] + df_c['z_Pay'] + df_c['z_Hurst'] + df_c['z_Toxic']) / 6

    # --- SAUVEGARDE ---
    today = datetime.now().date()
    try:
        with engine.begin() as conn:
            conn.execute(text('DELETE FROM fundamental_metrics WHERE "Date" = :d'), {"d": today})
            conn.execute(text('DELETE FROM qmj_scores WHERE "Date" = :d'), {"d": today})

            # Sauvegarde métriques brutes
            df_m = df_c.rename(columns={'Symbol':'Ticker', 'Safe_Hurst':'Hurst', 'Safe_Toxic':'Toxic_correl'})
            df_m['Date'] = today
            df_m[['Date','Ticker','Prof_GPOA','Prof_ROE','Prof_ROA','Prof_CFOA','Prof_Accruals','Grow_Earn','Grow_Rev','Safe_Beta','Safe_IVol','Safe_Lev','Hurst','Toxic_correl','Pay_Yield','Pay_Ratio']].to_sql('fundamental_metrics', conn, if_exists='append', index=False)

            # Sauvegarde scores (6 piliers)
            df_s = df_c.rename(columns={'Symbol':'Ticker', 'z_Prof':'z_Profitability', 'z_Grow':'z_Growth', 'z_Safe':'z_Safety', 'z_Pay':'z_Payout'})
            df_s['Date'] = today
            df_s[['Date','Ticker','Quality_Score','z_Profitability','z_Growth','z_Safety','z_Payout','z_Hurst','z_Toxic','Price']].to_sql('qmj_scores', conn, if_exists='append', index=False)

        print(f"\n✅ Succès : Modèle 6-Piliers mis à jour pour le {today}.")
    except Exception as e: print(f"❌ Erreur SQL: {e}")

    return df_c

if __name__ == "__main__":
    df_result = main()
