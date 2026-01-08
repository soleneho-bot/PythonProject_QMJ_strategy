import pandas as pd
from sqlalchemy import create_engine, text

# 1. Configuration
CSV_FILE = "strategy_performance_basis_rows.csv"

# LE BON LIEN (Celui utilisé par ton app.py et update_data.py)
DB_URL = "postgresql://postgres.hcuzqfwmddnekcefjfky:222PROJECTSVSJ222@aws-1-eu-west-1.pooler.supabase.com:6543/postgres"

# 2. Lecture du CSV
print(f"📖 Lecture de {CSV_FILE}...")
try:
    df = pd.read_csv(CSV_FILE)
    print(f"✅ Fichier chargé : {len(df)} lignes.")
except FileNotFoundError:
    print("❌ ERREUR : Le fichier CSV est introuvable. Vérifie qu'il est dans le même dossier.")
    exit()

# 3. Envoi vers Supabase (Base 'hcuz...')
print("🚀 Envoi vers la BONNE base Supabase...")
try:
    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        # On remplace la table si elle existe
        df.to_sql("strategy_performance_basis_rows", conn, if_exists='replace', index=False)
    print("✅ Succès ! La table 'strategy_performance_basis_rows' est en ligne.")
except Exception as e:
    print(f"❌ Erreur SQL : {e}")