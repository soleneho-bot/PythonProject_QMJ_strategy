import pandas as pd
import numpy as np
from scipy import stats

def calc_beta(y, x):
    """Calcul du coefficient Beta (pente de la régression)."""
    if len(y) < 60 or x.empty:
        return np.nan
    try:
        # On aligne les dates entre le titre et le marché
        common_idx = y.index.intersection(x.index)
        if len(common_idx) < 60: return np.nan
        return stats.linregress(x.loc[common_idx], y.loc[common_idx])[0]
    except:
        return np.nan

def calc_idio_vol(y, x):
    """Calcul de la volatilité idiosyncrasique (écart-type des résidus)."""
    if len(y) < 60 or x.empty:
        return np.nan
    try:
        common_idx = y.index.intersection(x.index)
        slope, intercept, _, _, _ = stats.linregress(x.loc[common_idx], y.loc[common_idx])
        residuals = y.loc[common_idx] - (intercept + slope * x.loc[common_idx])
        return residuals.std() * np.sqrt(252) # Annualisation
    except:
        return np.nan

def z_rank_group(series):
    """Calcule le Z-score basé sur le rang (robuste aux outliers)."""
    if series.count() < 2:
        return pd.Series(0, index=series.index)
    r = series.rank(method='average')
    return (r - r.mean()) / r.std()

def compute_quality_scores(df):
    """
    Prend un DataFrame de métriques brutes et calcule les scores QMJ finaux.
    """
    df_c = df.copy()
    
    # Remplacement des NaN par la médiane pour pouvoir calculer les scores
    for col in df_c.select_dtypes(include=np.number).columns:
        df_c[col] = df_c[col].fillna(df_c[col].median())

    # Définition des groupes de taille (Big vs Small)
    if 'MarketCap' in df_c.columns and df_c['MarketCap'].notna().sum() > 0:
        median_cap = df_c['MarketCap'].median()
        df_c['Size_Group'] = np.where(df_c['MarketCap'] >= median_cap, 'Big', 'Small')
    else:
        df_c['Size_Group'] = 'All'

    # 1. Score Profitabilité
    df_c['z_Prof'] = (
        df_c.groupby('Size_Group')['Prof_GPOA'].transform(z_rank_group) +
        df_c.groupby('Size_Group')['Prof_ROE'].transform(z_rank_group) +
        df_c.groupby('Size_Group')['Prof_ROA'].transform(z_rank_group) +
        df_c.groupby('Size_Group')['Prof_CFOA'].transform(z_rank_group) +
        (df_c.groupby('Size_Group')['Prof_Accruals'].transform(z_rank_group) * -1)
    ) / 5

    # 2. Score Croissance
    df_c['z_Grow'] = (
        df_c.groupby('Size_Group')['Grow_Earn'].transform(z_rank_group) +
        df_c.groupby('Size_Group')['Grow_Rev'].transform(z_rank_group)
    ) / 2

    # 3. Score Sécurité (Inverse du risque)
    df_c['z_Safe'] = (
        (df_c.groupby('Size_Group')['Safe_Beta'].transform(z_rank_group) * -1) +
        (df_c.groupby('Size_Group')['Safe_IVol'].transform(z_rank_group) * -1) +
        (df_c.groupby('Size_Group')['Safe_Lev'].transform(z_rank_group) * -1)
    ) / 3

    # 4. Score Payout
    df_c['z_Pay'] = (
        df_c.groupby('Size_Group')['Pay_Yield'].transform(z_rank_group) +
        df_c.groupby('Size_Group')['Pay_Ratio'].transform(z_rank_group)
    ) / 2

    # Score QUALITÉ FINAL (Moyenne des 4 piliers)
    df_c['Quality_Score'] = (df_c['z_Prof'] + df_c['z_Grow'] + df_c['z_Safe'] + df_c['z_Pay']) / 4
    
    return df_c
