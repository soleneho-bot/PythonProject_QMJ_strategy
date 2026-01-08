import os
import io
import json
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# 1. Chargement des variables d'environnement (le fichier .env)
load_dotenv()
DB_URL = os.getenv("DB_URL")

if not DB_URL:
    print("❌ Erreur : La variable DB_URL n'est pas définie dans le fichier .env")
    raise SystemExit

# 2. Création de l'engine SQLAlchemy
engine = create_engine(DB_URL)

def init_db():
    """Crée ou met à jour les tables de la base de données."""
    with engine.connect() as conn:
        # Table PRIX
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS prices (
                "Date" TIMESTAMP,
                "Ticker" TEXT,
                "Open" DOUBLE PRECISION,
                "High" DOUBLE PRECISION,
                "Low" DOUBLE PRECISION,
                "Close" DOUBLE PRECISION,
                "Volume" BIGINT,
                PRIMARY KEY ("Date", "Ticker")
            );
        '''))

        # Table SCORES QMJ
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS qmj_scores (
                "Date" DATE,
                "Ticker" TEXT,
                "Quality_Score" DOUBLE PRECISION,
                "z_Profitability" DOUBLE PRECISION,
                "z_Growth" DOUBLE PRECISION,
                "z_Safety" DOUBLE PRECISION,
                "z_Payout" DOUBLE PRECISION,
                "Price" DOUBLE PRECISION,
                PRIMARY KEY ("Date", "Ticker")
            );
        '''))

        # Table Financial Cache
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS financial_cache (
                "Ticker" TEXT,
                "Ticker_YF" TEXT,
                "Last_Updated" TIMESTAMP,
                "Financials" JSON,
                "Balance_Sheet" JSON,
                "Cashflow" JSON,
                "Info" JSON,
                PRIMARY KEY ("Ticker", "Last_Updated")
            );
        '''))

        # Table de métriques fondamentales
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS fundamental_metrics (
                "Date" DATE,
                "Ticker" TEXT,
                "Prof_GPOA" DOUBLE PRECISION,
                "Prof_ROE" DOUBLE PRECISION,
                "Prof_ROA" DOUBLE PRECISION,
                "Prof_CFOA" DOUBLE PRECISION,
                "Prof_Accruals" DOUBLE PRECISION,
                "Grow_Earn" DOUBLE PRECISION,
                "Grow_Rev" DOUBLE PRECISION,
                "Safe_Beta" DOUBLE PRECISION,
                "Safe_IVol" DOUBLE PRECISION,
                "Safe_Lev" DOUBLE PRECISION,
                "Pay_Yield" DOUBLE PRECISION,
                "Pay_Ratio" DOUBLE PRECISION,
                PRIMARY KEY ("Date", "Ticker")
            );
        '''))
        conn.commit()
    print("✅ Base de données initialisée avec succès.")

def load_df_from_json(field) -> pd.DataFrame:
    """Helper pour convertir les champs JSON de la DB en DataFrame."""
    if not field:
        return pd.DataFrame()
    try:
        if isinstance(field, str):
            return pd.read_json(io.StringIO(field))
        else:
            return pd.read_json(io.StringIO(json.dumps(field)))
    except Exception as e:
        print(f"[⚠️ Error] Impossible de parser le JSON : {e}")
        return pd.DataFrame()
