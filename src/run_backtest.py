import io
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# ============================================================
# 0. CONFIGURATION & CONNEXION
# ============================================================
DB_URL = "postgresql://postgres.zqtsrmdawsurenhddzku:222JULES2026!@aws-1-eu-central-1.pooler.supabase.com:6543/postgres"

try:
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT NOW()")).scalar()
    print(f"✅ Connecté à Supabase. Heure serveur : {res}")
except Exception as e:
    print(f"❌ Erreur de connexion : {e}")
    raise SystemExit

# ============================================================
# 1. OUTILS TICKERS & SCHEMA DB
# ============================================================
def canonical_to_yf(ticker: str) -> str:
    return ticker.replace('.', '-')

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(response.text))
        return tables[0]['Symbol'].tolist()
    except Exception as e:
        print(f"❌ Erreur S&P500 : {e}")
        return []

def init_db():
    """Crée les tables de manière atomique."""
    with engine.begin() as conn:
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS prices (
                "Date" TIMESTAMP, "Ticker" TEXT, "Open" DOUBLE PRECISION,
                "High" DOUBLE PRECISION, "Low" DOUBLE PRECISION,
                "Close" DOUBLE PRECISION, "Volume" BIGINT,
                PRIMARY KEY ("Date", "Ticker")
            );
        '''))
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS financial_cache (
                "Ticker" TEXT, "Ticker_YF" TEXT, "Last_Updated" TIMESTAMP,
                "Financials" JSON, "Balance_Sheet" JSON, "Cashflow" JSON, "Info" JSON,
                PRIMARY KEY ("Ticker", "Last_Updated")
            );
        '''))
    print("✅ Tables SQL (prices & cache) créées et validées.")

# ============================================================
# 2. CALCULS ACADÉMIQUES (SCRIPT 2)
# ============================================================
def calc_beta(y, x):
    if len(y) < 60: return np.nan
    return stats.linregress(x.loc[y.index], y)[0]

def calc_idio_vol(y, x):
    if len(y) < 60: return np.nan
    slope, intercept, _, _, _ = stats.linregress(x.loc[y.index], y)
    residuals = y - (intercept + slope * x.loc[y.index])
    return residuals.std() * np.sqrt(252)

def get_latest_series_before(df, asof_date, lag_days=90):
    if df is None or df.empty: return pd.Series(dtype='float64')
    cutoff = pd.to_datetime(asof_date) - pd.Timedelta(days=lag_days)
    eligible = [c for c in df.columns if pd.to_datetime(c) <= cutoff]
    return df[max(eligible)] if eligible else pd.Series(dtype='float64')

def compute_growth_3y_from_fin(fin_df, asof_date, lag_days=90):
    if fin_df is None or fin_df.empty: return np.nan
    cutoff = pd.to_datetime(asof_date) - pd.Timedelta(days=lag_days)
    cols_available = sorted([c for c in fin_df.columns if pd.to_datetime(c) <= cutoff])
    if len(cols_available) < 4: return np.nan
    ni_idx = [x for x in fin_df.index if 'Net Income' in str(x)]
    if not ni_idx: return np.nan
    ni_row = fin_df.loc[ni_idx[0], cols_available[-4:]]
    if ni_row.iloc[0] == 0 or pd.isna(ni_row.iloc[0]): return np.nan
    return (ni_row.iloc[-1] - ni_row.iloc[0]) / abs(ni_row.iloc[0])

# ============================================================
# 3. MOTEUR DE DONNÉES
# ============================================================
def update_prices_smart(tickers):
    start_history = datetime(2023, 1, 1)
    print(f"\n--- MAJ PRIX DEPUIS {start_history.date()} ---")
    ticker_map = {canonical_to_yf(t): t for t in tickers}
    ticker_map.update({'SPY': 'SPY'})

    new_data = yf.download(list(ticker_map.keys()), start=start_history, interval="1d", group_by='ticker', auto_adjust=True)
    data_to_insert = []
    for yf_t, cano in ticker_map.items():
        df_t = new_data[yf_t].copy() if len(ticker_map) > 1 else new_data.copy()
        df_t = df_t.dropna().reset_index()
        if df_t.empty: continue
        df_t['Ticker'] = cano
        df_t = df_t.rename(columns={c: c.capitalize() for c in df_t.columns if c.lower() in ['date','open','high','low','close','volume']})
        data_to_insert.append(df_t[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']])

    if data_to_insert:
        with engine.begin() as conn:
            conn.execute(text('DELETE FROM prices'))
            pd.concat(data_to_insert).to_sql('prices', conn, if_exists='append', index=False, method='multi')

def get_financial_data(ticker_canonical):
    query = text('SELECT * FROM financial_cache WHERE "Ticker" = :t ORDER BY "Last_Updated" DESC LIMIT 1')
    with engine.connect() as conn:
        res = conn.execute(query, {"t": ticker_canonical}).first()
    if res:
        def to_df(f): return pd.read_json(io.StringIO(json.dumps(f))) if isinstance(f, dict) else pd.read_json(io.StringIO(f))
        return to_df(res.Financials), to_df(res.Balance_Sheet), to_df(res.Cashflow), (json.loads(res.Info) if isinstance(res.Info, str) else res.Info)
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

def compute_fundamentals_at_date(ticker, fin, bs, cf, info, price_mat, ret_mat, asof_date):
    px_s = price_mat[ticker].ffill()
    px_at = px_s[px_s.index <= asof_date].iloc[-1] if not px_s[px_s.index <= asof_date].empty else np.nan
    curr_fin = get_latest_series_before(fin, asof_date)
    curr_bs = get_latest_series_before(bs, asof_date)
    curr_cf = get_latest_series_before(cf, asof_date)

    assets = curr_bs.get('Total Assets', info.get('totalAssets', np.nan))
    equity = curr_bs.get('Stockholders Equity', info.get('totalStockholderEquity', np.nan))
    ni = curr_fin.get('Net Income', info.get('netIncomeToCommon', np.nan))
    gp = curr_fin.get('Gross Profit', info.get('grossProfits', np.nan))
    ocf = curr_cf.get('Operating Cash Flow', info.get('operatingCashflow', np.nan))
    debt = curr_bs.get('Total Debt', info.get('totalDebt', np.nan))

    beta = ivol = np.nan
    if ticker in ret_mat.columns:
        window = ret_mat.loc[ret_mat.index <= asof_date].tail(252)
        if len(window) >= 60:
            y, x = window[ticker].dropna(), ret_mat['SPY'].loc[window.index].dropna()
            common = y.index.intersection(x.index)
            if len(common) >= 60:
                beta = calc_beta(y.loc[common], x.loc[common])
                ivol = calc_idio_vol(y.loc[common], x.loc[common])

    return {
        'Ticker': ticker, 'Date': asof_date.date(), 'Price': px_at,
        'Prof_GPOA': gp/assets if assets > 0 else np.nan,
        'Prof_ROE': ni/equity if equity > 0 else np.nan,
        'Prof_ROA': ni/assets if assets > 0 else np.nan,
        'Prof_CFOA': ocf/assets if assets > 0 else np.nan,
        'Prof_Accruals': (ni - ocf)/assets if assets > 0 else np.nan,
        'Grow_Earn': compute_growth_3y_from_fin(fin, asof_date),
        'Safe_Beta': beta, 'Safe_IVol': ivol, 'Safe_Lev': debt/assets if assets > 0 else np.nan
    }

# ============================================================
# 4. BACKTEST & EXPORT MENSUEL
# ============================================================
def run_backtest_logic(df_scores, ret_mat, all_tickers):
    strat_ret = pd.Series(0.0, index=ret_mat.index)
    weights_matrix = pd.DataFrame(0.0, index=ret_mat.index, columns=all_tickers)
    dates = sorted(df_scores['Date'].unique())
    for i, d in enumerate(dates):
        cross = df_scores[df_scores['Date'] == d].set_index('Ticker')
        n_sel = max(5, int(len(cross) * 0.3))
        top = cross.sort_values('Quality_Score', ascending=False).head(n_sel).index
        bot = cross.sort_values('Quality_Score', ascending=False).tail(n_sel).index

        current_weights = pd.Series(0.0, index=all_tickers)
        current_weights.loc[top] = 0.5 / n_sel
        current_weights.loc[bot] = -0.5 / n_sel

        end_date = dates[i+1] if i < len(dates)-1 else ret_mat.index[-1]
        mask = (ret_mat.index > pd.to_datetime(d)) & (ret_mat.index <= pd.to_datetime(end_date))
        for dt in ret_mat.index[mask]:
            strat_ret.loc[dt] = (ret_mat.loc[dt].reindex(all_tickers).fillna(0) * current_weights).sum()
            weights_matrix.loc[dt] = current_weights
    return strat_ret, weights_matrix

def export_to_supabase_monthly(strat_ret, bench_ret, weights_df):
    start_cum_date = pd.Timestamp('2024-01-03')
    df_perf = pd.DataFrame({"Strategy_Return": strat_ret, "Benchmark_Return": bench_ret})
    df_perf = df_perf[df_perf.index >= start_cum_date].fillna(0)
    df_perf["Cumulative_Strategy"] = (1 + df_perf["Strategy_Return"]).cumprod()
    df_perf["Cumulative_Benchmark"] = (1 + df_perf["Benchmark_Return"]).cumprod()
    df_perf["Cumulative_Strategy"] /= df_perf["Cumulative_Strategy"].iloc[0]
    df_perf["Cumulative_Benchmark"] /= df_perf["Cumulative_Benchmark"].iloc[0]

    final_output = pd.concat([df_perf, weights_df[weights_df.index >= start_cum_date]], axis=1)
    with engine.begin() as conn:
        conn.execute(text('DROP TABLE IF EXISTS strategy_performance_monthly'))
        final_output.reset_index().rename(columns={'index':'Date'}).to_sql('strategy_performance_monthly', conn, index=False)
    print("✅ Exportation 'strategy_performance_monthly' terminée.")
    return final_output

# ============================================================
# 5. MAIN
# ============================================================
def main():
    init_db()
    universe = get_sp500_tickers()[:100]
    update_prices_smart(universe)

    df_px = pd.read_sql('SELECT "Date", "Ticker", "Close" FROM prices', engine).pivot(index='Date', columns='Ticker', values='Close')
    df_ret = df_px.pct_change().dropna(how='all')

    start_backtest = datetime(2024, 1, 1).date()
    reb_dates = df_px[df_px.index.date >= start_backtest].index.to_series().resample('ME').max().tolist()

    f_cache = {t: get_financial_data(t) for t in universe}
    all_rows = []

    print(f"--- CALCUL MENSUEL QMJ ---")
    for d in reb_dates:
        print(f"[{d.date()}]", end=" ", flush=True)
        cs_data = [compute_fundamentals_at_date(t, *f_cache[t], df_px, df_ret, d) for t in universe if t in f_cache]
        df_cs = pd.DataFrame(cs_data).dropna(subset=['Price'])
        if df_cs.empty: continue

        for col in df_cs.select_dtypes(include=np.number).columns:
            df_cs[col] = df_cs[col].fillna(df_cs[col].median())

        z = lambda s: (s.rank() - s.rank().mean()) / s.rank().std() if s.count() > 1 else 0
        df_cs['z_Prof'] = (z(df_cs['Prof_GPOA']) + z(df_cs['Prof_ROE']) + z(df_cs['Prof_ROA']) + z(df_cs['Prof_CFOA']) - z(df_cs['Prof_Accruals'])) / 5
        df_cs['z_Grow'] = z(df_cs['Grow_Earn'])
        df_cs['z_Safe'] = (-z(df_cs['Safe_Beta']) - z(df_cs['Safe_IVol']) - z(df_cs['Safe_Lev'])) / 3
        df_cs['Quality_Score'] = (df_cs['z_Prof'] + df_cs['z_Grow'] + df_cs['z_Safe']) / 3
        all_rows.append(df_cs)

    full_scores = pd.concat(all_rows)
    strat_ret, w_mat = run_backtest_logic(full_scores, df_ret, universe)
    final_df = export_to_supabase_monthly(strat_ret, df_ret['SPY'], w_mat)

    plt.figure(figsize=(10, 5))
    plt.plot(final_df["Cumulative_Strategy"], label='QMJ Strat', color='green')
    plt.plot(final_df["Cumulative_Benchmark"], label='S&P 500', color='gray', alpha=0.5)
    plt.title("Performance Mensuelle (Base 1 au 03/01/2024)")
    plt.legend(); plt.show()

if __name__ == "__main__":
    main()