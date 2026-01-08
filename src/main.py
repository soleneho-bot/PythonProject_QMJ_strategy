import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text

# Import de nos propres modules
from database import engine, init_db
import data_fetcher
import engine as qmj_engine  # renommé pour éviter conflit avec sqlalchemy engine

def main():
    # A. Initialisation
    init_db()

    # B. Récupération de l'univers (S&P 500)
    tickers = data_fetcher.get_sp500_tickers()
    print(f"🚀 Univers chargé : {len(tickers)} tickers.")

    # C. Mise à jour des prix
    data_fetcher.update_prices_smart(tickers)

    # D. Chargement des prix pour calculs techniques (Beta/Vol)
    print("-> Chargement des prix depuis la DB...")
    query_prices = 'SELECT "Date", "Ticker", "Close" FROM prices WHERE "Date" >= NOW() - INTERVAL \'730 days\''
    prices_raw = pd.read_sql(query_prices, engine, parse_dates=['Date'])
    
    if prices_raw.empty:
        print("❌ Aucun prix en base. Arrêt.")
        return

    prices_df = prices_raw.pivot(index='Date', columns='Ticker', values='Close')
    returns_df = prices_df.pct_change().dropna()
    market_returns = returns_df['SPY'] if 'SPY' in returns_df.columns else pd.Series(dtype=float)

    # E. Analyse Fondamentale
    print("\n--- 2. ANALYSE FONDAMENTALE ---")
    fundamental_data = []
    last_prices = prices_df.ffill().iloc[-1].to_dict()

    for i, ticker in enumerate(tickers):
        if ticker == 'SPY': continue
        if i % 20 == 0: print(f"Progression : {i}/{len(tickers)}")

        fin, bs, cf, info, is_cached = data_fetcher.get_financial_data(ticker)
        
        if fin.empty or not info: continue

        try:
            # Extraction des variables clés
            curr_fin = fin.iloc[:, 0]
            curr_bs = bs.iloc[:, 0]
            curr_cf = cf.iloc[:, 0]

            assets = curr_bs.get('Total Assets', info.get('totalAssets', np.nan))
            debt = curr_bs.get('Total Debt', info.get('totalDebt', np.nan))
            net_income = curr_fin.get('Net Income', info.get('netIncomeToCommon', np.nan))
            rev = curr_fin.get('Total Revenue', np.nan)
            ocf = curr_cf.get('Operating Cash Flow', info.get('operatingCashflow', np.nan))

            # Calcul des métriques brutes
            gpoa = (rev - curr_fin.get('Cost Of Revenue', 0)) / assets if assets else np.nan
            
            fundamental_data.append({
                'Symbol': ticker,
                'Price': last_prices.get(ticker, info.get('currentPrice', np.nan)),
                'MarketCap': info.get('marketCap', np.nan),
                'Prof_GPOA': gpoa,
                'Prof_ROE': info.get('returnOnEquity', np.nan),
                'Prof_ROA': info.get('returnOnAssets', np.nan),
                'Prof_CFOA': ocf / assets if assets else np.nan,
                'Prof_Accruals': (net_income - ocf) / assets if assets else np.nan,
                'Grow_Earn': info.get('earningsQuarterlyGrowth', np.nan),
                'Grow_Rev': info.get('revenueGrowth', np.nan),
                'Safe_Beta': qmj_engine.calc_beta(returns_df[ticker], market_returns) if ticker in returns_df else np.nan,
                'Safe_IVol': qmj_engine.calc_idio_vol(returns_df[ticker], market_returns) if ticker in returns_df else np.nan,
                'Safe_Lev': debt / assets if assets else np.nan,
                'Pay_Yield': info.get('dividendYield', 0),
                'Pay_Ratio': info.get('payoutRatio', 0)
            })
        except Exception as e:
            continue

    # F. Calcul des Scores via notre Engine
    raw_df = pd.DataFrame(fundamental_data)
    scored_df = qmj_engine.compute_quality_scores(raw_df)

    # G. Sauvegarde des résultats
    today = datetime.now().date()
    print(f"\n✅ Analyse terminée. Sauvegarde des scores pour le {today}...")
    
    final_to_save = scored_df[['Symbol', 'Quality_Score', 'z_Prof', 'z_Grow', 'z_Safe', 'z_Pay', 'Price']].copy()
    final_to_save['Date'] = today
    final_to_save = final_to_save.rename(columns={'Symbol': 'Ticker', 'z_Prof': 'z_Profitability', 'z_Grow': 'z_Growth', 'z_Safe': 'z_Safety', 'z_Pay': 'z_Payout'})

    with engine.begin() as conn:
        conn.execute(text('DELETE FROM qmj_scores WHERE "Date" = :d'), {"d": today})
        final_to_save.to_sql('qmj_scores', conn, if_exists='append', index=False, method='multi')

    # H. Affichage du TOP 5
    print("\n--- 🔥 TOP 5 QUALITY STOCKS ---")
    print(scored_df.sort_values('Quality_Score', ascending=False)[['Symbol', 'Quality_Score', 'Prof_GPOA', 'Safe_Beta']].head(5))

if __name__ == "__main__":
    main()
