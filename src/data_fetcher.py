import yfinance as yf
import pandas as pd
import requests
import io
import json
from datetime import datetime, timedelta
from sqlalchemy import text
from database import engine, load_df_from_json

def get_sp500_tickers():
    """Récupère la liste des tickers du S&P 500 depuis Wikipédia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(response.text))
        for t in tables:
            if 'Symbol' in t.columns:
                return t['Symbol'].values.tolist()
        return []
    except Exception as e:
        print(f"❌ Erreur récupération S&P500 : {e}")
        return []

def canonical_to_yf(ticker: str) -> str:
    return ticker.replace('.', '-')

def yf_to_canonical(yf_ticker: str) -> str:
    return yf_ticker.replace('-', '.')

def update_prices_smart(tickers_canonical):
    """Mise à jour incrémentale des prix dans la base de données."""
    print(f"\n--- 1. MISE À JOUR PRIX ({len(tickers_canonical)} tickers) ---")
    
    try:
        last_date_df = pd.read_sql('SELECT MAX("Date") as last_date FROM prices', engine)
        last_date_ts = last_date_df['last_date'].iloc[0]
    except:
        last_date_ts = None

    if last_date_ts:
        start_date = pd.to_datetime(last_date_ts) + timedelta(days=1)
        if start_date.date() > datetime.now().date():
            print("-> Prix déjà à jour.")
            return
    else:
        start_date = datetime.now() - timedelta(days=730)

    ticker_map = {canonical_to_yf(t): t for t in tickers_canonical}
    ticker_map.setdefault('SPY', 'SPY')
    tickers_dl = list(ticker_map.keys())

    new_data = yf.download(tickers_dl, start=start_date, group_by='ticker', auto_adjust=True)
    
    if new_data.empty: return

    data_to_insert = []
    for yf_ticker in tickers_dl:
        try:
            if isinstance(new_data.columns, pd.MultiIndex):
                if yf_ticker not in new_data.columns.levels[0]: continue
                df_t = new_data[yf_ticker].copy()
            else:
                df_t = new_data.copy()

            df_t = df_t.dropna().reset_index()
            canonical = ticker_map.get(yf_ticker, yf_to_canonical(yf_ticker))
            df_t['Ticker'] = canonical
            
            # Renommage propre pour SQL
            df_t = df_t.rename(columns={c: c.capitalize() for c in df_t.columns if c.lower() in ['open', 'high', 'low', 'close', 'volume', 'date']})
            data_to_insert.append(df_t[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']])
        except: continue

    if data_to_insert:
        pd.concat(data_to_insert).to_sql('prices', engine, if_exists='append', index=False, method='multi', chunksize=1000)
        print(f"✅ Prix mis à jour.")

def get_financial_data(ticker_canonical, ttl_days=30):
    """Récupère les états financiers (via cache DB ou Yahoo Finance)."""
    ticker_yf = canonical_to_yf(ticker_canonical)
    
    query = text('SELECT "Last_Updated", "Financials", "Balance_Sheet", "Cashflow", "Info" '
                 'FROM financial_cache WHERE "Ticker" = :t ORDER BY "Last_Updated" DESC LIMIT 1')

    with engine.connect() as conn:
        result = conn.execute(query, {"t": ticker_canonical}).first()

    if result:
        last_update, fin_raw, bs_raw, cf_raw, info_raw = result
        if (datetime.now() - pd.to_datetime(last_update).tz_localize(None)).days < ttl_days:
            return load_df_from_json(fin_raw), load_df_from_json(bs_raw), load_df_from_json(cf_raw), info_raw if isinstance(info_raw, dict) else json.loads(info_raw), True

    # Si pas de cache, on télécharge
    try:
        stock = yf.Ticker(ticker_yf)
        fin, bs, cf, info = stock.financials, stock.balance_sheet, stock.cashflow, stock.info
        
        with engine.begin() as conn:
            conn.execute(text('INSERT INTO financial_cache ("Ticker", "Ticker_YF", "Last_Updated", "Financials", "Balance_Sheet", "Cashflow", "Info") '
                             'VALUES (:t, :tyf, :d, :f, :b, :c, :i)'),
                         {"t": ticker_canonical, "tyf": ticker_yf, "d": datetime.now(), 
                          "f": fin.to_json(date_format='iso'), "b": bs.to_json(date_format='iso'), 
                          "c": cf.to_json(date_format='iso'), "i": json.dumps(info)})
        return fin, bs, cf, info, False
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, False
