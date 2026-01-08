import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

# ==============================================================================
# 1. CONFIGURATION ET CONNEXIONS
# ==============================================================================
st.set_page_config(page_title="QMJ Terminal", layout="wide", page_icon="🦅")

# --- BASES DE DONNÉES ---
DB_URL_BACKTEST = "postgresql://postgres.zqtsrmdawsurenhddzku:222JULES2026!@aws-1-eu-central-1.pooler.supabase.com:6543/postgres"
DB_URL_LIVE = "postgresql://postgres.hcuzqfwmddnekcefjfky:222PROJECTSVSJ222@aws-1-eu-west-1.pooler.supabase.com:6543/postgres"

# --- STYLE CSS ---
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif;}
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. FONCTIONS DE CHARGEMENT DE DONNÉES
# ==============================================================================

@st.cache_data(ttl=3600)
def get_live_data():
    """Récupère les scores actuels depuis la base LIVE (Table: qmj_scores)"""
    try:
        engine = create_engine(DB_URL_LIVE)
        # On récupère tout le contenu de qmj_scores
        query = 'SELECT * FROM "qmj_scores"'
        
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
            
        if not df.empty:
            # Conversion date
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                # On ne garde que la date la plus récente pour le Live Terminal
                last_date = df['Date'].max()
                df = df[df['Date'] == last_date]
            
            # Gestion MarketCap (si absent, on met une valeur par défaut pour l'affichage)
            if 'MarketCap' not in df.columns:
                df['MarketCap'] = 1e9 
                
        return df
    except Exception as e:
        st.error(f"Erreur connexion Live DB: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_backtest_data():
    """
    Récupère et fusionne les deux backtests depuis la base BACKTEST.
    - strategy_performance : Notre Stratégie
    - strategy_performance_monthly : Stratégie AQR Originale
    """
    try:
        engine = create_engine(DB_URL_BACKTEST)
        
        # 1. Charger NOTRE stratégie
        # On sélectionne les colonnes principales de perf
        cols_perf = '"Date", "Strategy_Return", "Benchmark_Return", "Cumulative_Strategy", "Cumulative_Benchmark"'
        q_our = f'SELECT {cols_perf} FROM "strategy_performance"'
        
        # 2. Charger AQR (Originale)
        q_aqr = f'SELECT {cols_perf} FROM "strategy_performance_monthly"'
        
        with engine.connect() as conn:
            df_our = pd.read_sql(text(q_our), conn)
            df_aqr = pd.read_sql(text(q_aqr), conn)
            
        # Conversion Dates
        df_our['Date'] = pd.to_datetime(df_our['Date'])
        df_aqr['Date'] = pd.to_datetime(df_aqr['Date'])
        
        # Renommage pour fusion propre
        # Our Strat -> Suffixe _Our
        df_our = df_our.rename(columns={
            'Strategy_Return': 'Ret_Our',
            'Cumulative_Strategy': 'Cum_Our'
        })
        
        # AQR Strat -> Suffixe _AQR
        df_aqr = df_aqr.rename(columns={
            'Strategy_Return': 'Ret_AQR',
            'Cumulative_Strategy': 'Cum_AQR'
        })
        
        # Fusion sur la Date
        # On garde aussi Cumulative_Benchmark (qui devrait être le même, on prend celui de "Our")
        df_merged = pd.merge(
            df_our[['Date', 'Ret_Our', 'Cum_Our', 'Cumulative_Benchmark']], 
            df_aqr[['Date', 'Ret_AQR', 'Cum_AQR']], 
            on='Date', 
            how='inner'
        )
        
        return df_merged.sort_values('Date')
        
    except Exception as e:
        st.error(f"Erreur connexion Backtest DB: {e}")
        return pd.DataFrame()

# ==============================================================================
# 3. INTERFACE STREAMLIT
# ==============================================================================

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=50)
    st.title("NAVIGATION")
    page = st.radio("Menu", [
        "1. Executive Summary", 
        "2. Stratégie AQR (Théorie)", 
        "3. Nos Innovations", 
        "4. Backtest (Comparatif)", 
        "5. Live Strategy"
    ])
    st.markdown("---")
    st.caption("Données connectées à Supabase 🟢")

# ==============================================================================
# PAGE 1 : EXECUTIVE SUMMARY (NOUVEAU)
# ==============================================================================
if page == "1. Executive Summary":
    st.subheader("Executive Summary")
    st.caption("Performance de la Stratégie Propriétaire (QMJ + Hurst + Toxic)")
    
    df_bt = get_backtest_data()
    
    if df_bt.empty:
        st.warning("Données de performance non disponibles.")
    else:
        # --- CALCUL DES KPIs CLÉS ---
        last_row = df_bt.iloc[-1]
        
        # Rendements
        total_ret = (last_row['Cum_Our'] - 1) * 100
        
        # Risque
        vol = df_bt['Ret_Our'].std() * np.sqrt(252) * 100
        sharpe = (total_ret / vol) if vol != 0 else 0
        
        # Max Drawdown
        cum = df_bt['Cum_Our']
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min() * 100

        # --- AFFICHAGE EN CARTES (4 Colonnes) ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rendement Total", f"{total_ret:.1f}%") # Comparaison SPY retirée
        col2.metric("Sharpe Ratio", f"{sharpe:.2f}", "Risk-Adj") # CAGR retiré
        col3.metric("Volatilité", f"{vol:.1f}%", delta_color="inverse")
        col4.metric("Max Drawdown", f"{max_dd:.1f}%", delta_color="inverse")
        
        st.markdown("---")
        
       # --- GRAPHIQUE HERO (Performance Historique) ---
        st.subheader("Performance Historique") # Renommé
        
        # On ne garde que 'Notre Stratégie'
        df_chart = df_bt[['Date', 'Cum_Our']].copy()
        df_chart.columns = ['Date', 'Notre Stratégie']
        df_chart = df_chart.melt('Date', var_name='Asset', value_name='Value')
        
        fig = px.line(
            df_chart, x='Date', y='Value', color='Asset',
            color_discrete_map={'Notre Stratégie': '#00CC96'},
            template="plotly_dark", height=450
        )
        
        # Styling "Moderne" : Ligne plus épaisse, remplissage subtil, layout épuré
        fig.update_traces(
            line=dict(width=3), 
            fill='tozeroy', 
            fillcolor='rgba(0, 204, 150, 0.15)' # Effet de transparence moderne
        )
        
        fig.update_layout(
            yaxis=dict(
                range=[0.8, 1.4], # Range imposé
                showgrid=True, 
                gridcolor='rgba(255, 255, 255, 0.05)', # Grille très discrète
                zeroline=False
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False
            ),
            plot_bgcolor='rgba(0,0,0,0)', # Fond transparent pour s'intégrer à Streamlit
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h", 
                y=1.1, 
                x=0, 
                title=None, # Retire le titre de la légende inutile
                bgcolor='rgba(0,0,0,0)'
            ),
            hovermode="x unified", # Tooltip moderne qui suit la souris
            margin=dict(l=0, r=0, t=20, b=0) # Optimisation de l'espace
        )
        
        st.plotly_chart(fig, use_container_width=True)


        # --- SOUS-ANALYSE 1 (Drawdown & Distribution) ---
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🌊 Profil de Drawdown")
            st.caption("Profondeur des pertes par rapport aux plus hauts historiques.")
            df_dd_chart = pd.DataFrame({'Date': df_bt['Date'], 'Drawdown': dd})
            fig_dd = px.area(df_dd_chart, x='Date', y='Drawdown', color_discrete_sequence=['#EF553B'], template="plotly_dark", height=250)
            fig_dd.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_dd, use_container_width=True)
        
        with c2:
            st.markdown("#### 📊 Distribution des Rendements")
            st.caption("Fréquence des rendements journaliers (Aymétrie).")
            fig_hist = px.histogram(df_bt, x='Ret_Our', nbins=50, color_discrete_sequence=['#00CC96'], template="plotly_dark", height=250)
            fig_hist.update_layout(xaxis_title="Rendement Journalier", yaxis_title="Fréquence")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # --- SOUS-ANALYSE 2 (Nouveaux Graphiques Avancés) ---
        st.subheader("📉 Analyse Dynamique (Risque & Rendement)")
        c3, c4 = st.columns(2)

        with c3:
            st.markdown("#### 🎢 Volatilité Roulante (30 Jours)")
            st.caption("Nervosité de la stratégie dans le temps.")
            # Calcul Volatilité Roulante
            roll_vol = df_bt['Ret_Our'].rolling(30).std() * np.sqrt(252) * 100
            df_vol = pd.DataFrame({'Date': df_bt['Date'], 'Volatilité': roll_vol})
            
            fig_vol = px.line(df_vol, x='Date', y='Volatilité', color_discrete_sequence=['#FFA15A'], template="plotly_dark", height=250)
            fig_vol.update_layout(yaxis_title="Volatilité (%)")
            st.plotly_chart(fig_vol, use_container_width=True)

        with c4:
            st.markdown("#### 💎 Rolling Sharpe Ratio (6 Mois)")
            st.caption("Efficacité du rendement ajusté du risque (Fenêtre 126 jours).")
            # Calcul Sharpe Roulant (Rendement Annualisé / Volatilité Annualisée)
            window = 126 # ~6 mois de trading
            roll_ret = df_bt['Ret_Our'].rolling(window).mean() * 252
            roll_std = df_bt['Ret_Our'].rolling(window).std() * np.sqrt(252)
            roll_sharpe = roll_ret / roll_std
            
            df_sharpe = pd.DataFrame({'Date': df_bt['Date'], 'Sharpe': roll_sharpe})
            
            fig_sharpe = px.area(df_sharpe, x='Date', y='Sharpe', color_discrete_sequence=['#636EFA'], template="plotly_dark", height=250)
            # Ajout ligne 0 pour voir si le Sharpe passe négatif
            fig_sharpe.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)
            st.plotly_chart(fig_sharpe, use_container_width=True)

# ==============================================================================
# PAGE 2 : STRATÉGIE AQR (THÉORIE APPROFONDIE)
# ==============================================================================
elif page == "2. Stratégie AQR (Théorie)":
    st.subheader("La Recherche AQR : Quality Minus Junk")
    st.caption("Basé sur le papier académique de Clifford Asness, Andrea Frazzini et Lasse Pedersen (2013).")

    # --- 1. LE CONCEPT CENTRAL ---
    st.write("---")
    c_text, c_formula = st.columns([1.5, 1])
    
    with c_text:
        st.subheader("L'Anomalie Qualité")
        st.markdown("""
        La théorie financière classique suggère que **plus de risque = plus de rendement**.
        Cependant, AQR a démontré une anomalie majeure : les actions de **Haute Qualité** (Sûres, Profitables) surperforment historiquement les actions "Junk" (Risquées, Non-profitables) sur une base ajustée du risque.
        
        La stratégie **QMJ** consiste à :
        * 🟢 Acheter (**Long**) le top 30% des actions "Qualité".
        * 🔴 Vendre (**Short**) le top 30% des actions "Junk".
        """)

    with c_formula:
        st.info("🧮 **La Formule du Score QMJ**")
        st.latex(r'''
        Quality = z(Profit) + z(Growth) + z(Safety) + z(Payout)
        ''')
        st.caption("*Chaque variable est convertie en Z-Score (rang normalisé) pour être comparable.*")

    # --- 2. LES 4 PILIERS DÉTAILLÉS (SELON LE PAPIER) ---
    st.write("---")
    st.subheader("🏛️ Les 4 Piliers de la Qualité")
    st.markdown("Pour AQR, une entreprise de qualité se définit par ces caractéristiques précises :")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-top: 3px solid #00CC96;">
            <h4>💰 Profitabilité</h4>
            <div style="text-align: left; font-size: 0.9em; color: #aaa;">
            La capacité à générer des profits par rapport aux actifs.<br><br>
            <b>Métriques Clés :</b><br>
            • <b>GPOA</b> (Gross Profit / Assets)<br>
            • <b>ROE</b> (Return on Equity)<br>
            • <b>ROA</b> (Return on Assets)<br>
            • <b>CFOA</b> (Cash Flow / Assets)<br>
            • <b>Accruals</b> (Faibles)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="border-top: 3px solid #636EFA;">
            <h4>🚀 Croissance</h4>
            <div style="text-align: left; font-size: 0.9em; color: #aaa;">
            La variation de la profitabilité sur les 5 dernières années.<br><br>
            <b>Métriques Clés :</b><br>
            • <b>Δ Profitability</b><br>
            (Croissance du GPOA, ROE, etc.)<br>
            <br><i>"La qualité, c'est bien. Une qualité qui augmente, c'est mieux."</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="border-top: 3px solid #FFA15A;">
            <h4>🛡️ Sécurité</h4>
            <div style="text-align: left; font-size: 0.9em; color: #aaa;">
            La stabilité des résultats et la solidité financière.<br><br>
            <b>Métriques Clés :</b><br>
            • <b>Low Beta</b> (Faible sensibilité)<br>
            • <b>Low Leverage</b> (Faible dette)<br>
            • <b>Low Volatility</b> (ROAj, ROEj)<br>
            • <b>Low Credit Risk</b> (O-Score)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card" style="border-top: 3px solid #EF553B;">
            <h4>💸 Payout</h4>
            <div style="text-align: left; font-size: 0.9em; color: #aaa;">
            La redistribution de la richesse aux actionnaires.<br><br>
            <b>Métriques Clés :</b><br>
            • <b>Dividend Yield</b><br>
            • <b>Net Buybacks</b> (Rachats d'actions)<br>
            • <b>Net Equity Issuance</b> (Faible dilution)
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- 3. ILLUSTRATION DU CONCEPT (CONCEPTUAL CHART) ---
    st.write("---")
    st.subheader("📊 Performance Conceptuelle (Backtest AQR 1986-2012)")
    
    # Simulation stylisée des résultats du papier (Tableau 1 du PDF)
    # AQR montre un Alpha positif significatif
    
    viz_col1, viz_col2 = st.columns([2, 1])
    
    with viz_col1:
        # Création d'un graphique conceptuel
        years = np.arange(2000, 2025)
        # Simulation d'un spread croissant (Conceptuel)
        qmj_perf = np.cumsum(np.random.normal(0.05, 0.1, len(years))) + np.linspace(0, 2.5, len(years))
        mkt_perf = np.cumsum(np.random.normal(0.06, 0.15, len(years))) + np.linspace(0, 1.5, len(years)) # Market plus volatil
        
        df_concept = pd.DataFrame({'Année': years, 'Stratégie QMJ (Alpha)': qmj_perf, 'Junk Stocks': mkt_perf * 0.5})
        df_melt = df_concept.melt('Année', var_name='Type', value_name='Performance Cumulée')
        
        fig_concept = px.line(
            df_melt, x='Année', y='Performance Cumulée', color='Type',
            title="Illustration : Qualité vs Junk (Conceptuel)",
            color_discrete_map={'Stratégie QMJ (Alpha)': '#00CC96', 'Junk Stocks': '#EF553B'},
            template="plotly_dark", height=350
        )
        # Annotation
        fig_concept.add_annotation(x=2024, y=df_concept['Stratégie QMJ (Alpha)'].iloc[-1], text="Surperformance Structurelle", showarrow=True, arrowhead=1)
        st.plotly_chart(fig_concept, use_container_width=True)
    
    with viz_col2:
        st.info("💡 **Pourquoi ça marche ?**")
        st.markdown("""
        Selon le papier, deux explications coexistent :
        
        1.  **Risque mal évalué ?** Non. Les actions Qualité sont *moins* risquées lors des crises.
        2.  **Erreur comportementale :** Les investisseurs paient trop cher pour des actions "Loterie" (Junk) qui promettent beaucoup mais délivrent peu, et sous-estiment les entreprises "Ennuyeuses" mais rentables (Qualité).
        """)

# ==============================================================================
# PAGE 3 : INNOVATIONS (VERSION COMPLÈTE)
# ==============================================================================
elif page == "3. Nos Innovations":
    st.subheader("Nos Contributions : Au-delà d'AQR")
    st.caption("Comment dynamiser une stratégie fondamentale trop statique ?")

    # --- 1. LE PROBLÈME ---
    st.write("---")
    st.subheader("⚠️ Le Constat : Le Risque de 'Signal Périmé'")
    c_prob1, c_prob2 = st.columns([2, 1])
    
    with c_prob1:
        st.markdown("""
        Le modèle original d'AQR repose sur des données comptables (Bilan, Compte de résultat) qui ne changent que **trimestriellement**.
        
        **Le Danger :** Si une actualité brutale frappe une entreprise (ex: perte d'un procès, scandale), ses chiffres comptables restent "excellents" pendant encore 3 mois. Le modèle continue d'acheter l'action alors qu'elle s'effondre.
        
        👉 **Notre Objectif :** Intégrer des signaux à haute fréquence (quotidiens) pour réagir plus vite que le comptable.
        """)
    with c_prob2:
        st.info("💡 **Solution**\n\nAjouter des filtres techniques et comportementaux qui se mettent à jour chaque nuit.")

    # --- 2. LES SOLUTIONS IMPLÉMENTÉES (HURST & TOXIC) ---
    st.write("---")
    st.subheader("✅ Les 2 Piliers Ajoutés (Implémentés)")
    
    tab_hurst, tab_toxic = st.tabs(["1. Persistance (Hurst)", "2. Crédit Implicite (Toxic Correlation)"])
    
    with tab_hurst:
        col_h1, col_h2 = st.columns([1, 1])
        with col_h1:
            st.markdown("### 🌊 L'Exposant de Hurst")
            st.markdown("**Concept :** Mesure la mémoire à long terme d'une série temporelle. Il permet de savoir si un marché : est aléatoire / présente une tendance persistante / ou un comportement de retour à la moyenne")
            st.success("""
            * **H > 0.5 (Notre Cible) :** "Compounders". La tendance est persistante. La performance vient d'un avantage durable (Moat).
            * **H ≤ 0.5 (Exclu) :** Mouvement brownien (bruit) ou retour à la moyenne violent.
            """)
            st.caption("Référence : Benoit Mandelbrot (1963), 'The Variation of Certain Speculative Prices'.")
        with col_h2:
            st.warning("⚠️ **Limites :** Sensible à la fenêtre de calcul. Risque d'overtrading si H oscille autour de 0.5.")

    with tab_toxic:
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("### ☢️ Le Risque de Crédit Implicite")
            st.markdown("**Concept :** Utiliser la corrélation avec les Junk Bonds (ETF HYG) comme un détecteur de fumée.")
            st.error("""
            * Si une action est **trop corrélée** au High Yield, son "Equity" se comporte comme de la "Dette risquée".
            * C'est un signe avant-coureur de faillite ou de crise de liquidité.
            """)
            st.caption("Référence : Merton (1974), 'On the Pricing of Corporate Debt'.")
        with col_t2:
            st.markdown("**Implémentation :**")
            st.code("Toxic_Score = Correlation(Stock_Returns, HYG_Returns, window=60)")

    # --- 3. LE LABORATOIRE (PISTES EXPLORÉES) ---
    st.write("---")
    st.subheader("🧪 Le Laboratoire : Pistes Explorées & Non Retenues")
    st.caption("Nous avons testé et écarté plusieurs indicateurs pour garder le modèle robuste.")

    with st.expander("Voir les pistes abandonnées (Google Trends, Smart Money, etc.)"):
        st.markdown("""
        ### 1. Le Sentiment Spéculatif (Google Trends & GDELT)
        * **Hypothèse :** Un pic de recherche sur un ticker = Activité "Retail" spéculative (Signe de bulle/Junk).
        * **Référence :** Preis, Moat, & Stanley (2013).
        * **Pourquoi abandonné ?** Instabilité de l'API Google avec beaucoup d'errerurs et difficulté à normaliser les données "Hype" en temps réel. 
                                   Le volume de données GDELT est titanesque. Isoler le sentiment lié spécifiquement à un ticker 
                                  (et pas à une homonymie) demandait des modèles NLP (Traitement du Langage Naturel) trop lourds 
                                  pour un script quotidien.

        ### 2. Le Smart Money (Short Interest)
        * **Hypothèse :** Les vendeurs à découvert sont les investisseurs les mieux informés.
        * **Règle testée :** Exclure tout titre dont >15% du flottant est shorté (les titres massivement vendus à découvert par les hedge funds).
        * **Statut :** Le manque de Data. Les données de Short Interest gratuites sont souvent retardées de 15 jours, ce qui annulait l'intérêt d'un indicateur "dynamique".

        ### 3. Le Score Global de Synthèse (RVI + Pression)
        * **Idée :** Combiner Hype (Volume), Pression (Prix/Volume) et Tendance (MA50) en un seul score via une fonction `tanh`.
        * **Résultat :** Alternative aux "Google Trends" car, en bourse, un pic de volume précède souvent un mouvement de prix majeur. Trop complexe et redondant avec le Hurst Exponent.
        
        ### 4. Lottery Ticket 
        * **Idée :** Les actions ayant des hausses quotidiennes extrêmes sont spéculatives (Junk) et sous-performent.
        * **Référence :** Bali, Cakici, & Whitelaw (2011)
        * **Résultat :** Pénalité de Score : Plus le rendement max sur 21 jours est élevé, plus le score "Safety" diminue.
                    
        ### 5. Skin in the Game 
        * **Idée :** Les dirigeants n'achètent des actions que s'ils sont convaincus de la sous-évaluation ou de la croissance future.
        * **Référence :** N. Taleb 
        * **Résultat :** Bonus de Score : Un achat net des dirigeants sur 6 mois augmente le score "Quality" du titre.

        ### 6. Beneish M-Score 
        * **Idée :** Détecter les manipulations de profits avant qu'elles ne soient publiques.
        * **Résultat :** Filtre Binaire : Le score combine 8 ratios (croissance des ventes suspecte, dépréciation anormale, levier, etc.). Si M > -1.78, l'action est marquée "Red Flag" et exclue du portefeuille Long.
        """)

        

    # --- 4. TABLEAU RÉCAPITULATIF ---
    st.write("---")
    st.subheader("📋 Synthèse de l'Architecture Finale")
    
    data_archi = {
        "Indicateur": ["GPOA/ROE/ROA", "Hurst Exponent", "Toxic Correlation", "Beneish M-Score"],
        "Type": ["Fondamental (Lent)", "Technique (Rapide)", "Macro (Rapide)", "Forensic (Comptable)"],
        "Rôle": ["Sélectionner la Qualité", "Valider la Tendance", "Éviter le Crash", "Détecter la Fraude"],
        "Statut": ["✅ Actif (Cœur)", "✅ Actif (Filtre)", "✅ Actif (Risque)", "❌ Manuel (Audit)"]
    }
    st.table(pd.DataFrame(data_archi))

# --- PAGE 4 : BACKTEST COMPARATIF ---
elif page == "4. Backtest (Comparatif)":
    st.title("📈 Performance Historique")
    st.caption("Comparaison : Notre Stratégie vs AQR Original vs Benchmark (S&P 500)")
    
    df_bt = get_backtest_data()
    
    if df_bt.empty:
        st.warning("Aucune donnée de backtest trouvée. Vérifiez les tables 'strategy_performance' et 'strategy_performance_monthly'.")
    else:
        # --- CALCULS KPIs ---
        last_row = df_bt.iloc[-1]
        
        # Rendements Totaux
        ret_our = (last_row['Cum_Our'] - 1) * 100
        ret_aqr = (last_row['Cum_AQR'] - 1) * 100
        ret_bench = (last_row['Cumulative_Benchmark'] - 1) * 100
        
        # Volatilité (Annualisée)
        vol_our = df_bt['Ret_Our'].std() * np.sqrt(252) * 100 # Hypothèse données journalières (252)
        vol_aqr = df_bt['Ret_AQR'].std() * np.sqrt(252) * 100
        
        # Sharpe
        sharpe_our = ret_our / vol_our if vol_our != 0 else 0
        sharpe_aqr = ret_aqr / vol_aqr if vol_aqr != 0 else 0
        
        # --- AFFICHAGE KPIs ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Perf. Notre Stratégie", f"{ret_our:.1f}%", f"{ret_our - ret_aqr:.1f}% vs AQR")
        c2.metric("Perf. Benchmark", f"{ret_bench:.1f}%")
        c3.metric("Volatilité (Nôtre)", f"{vol_our:.1f}%", f"{vol_our - vol_aqr:.1f}% vs AQR", delta_color="inverse")
        c4.metric("Sharpe Ratio", f"{sharpe_our:.2f}", f"{sharpe_our - sharpe_aqr:.2f} vs AQR")
        
        st.markdown("---")
        
        # --- GRAPHIQUE PRINCIPAL ---
        st.subheader("Perfomance Historique")
        
        # Mise en forme pour Plotly
        df_chart = df_bt[['Date', 'Cum_Our', 'Cum_AQR', 'Cumulative_Benchmark']].melt('Date', var_name='Stratégie', value_name='Valeur')
        
        # Dictionnaire de noms propres pour la légende
        legend_map = {
            'Cum_Our': 'Notre Stratégie',
            'Cum_AQR': 'Stratégie AQR',
            'Cumulative_Benchmark': 'S&P 500'
        }
        df_chart['Stratégie'] = df_chart['Stratégie'].map(legend_map)
        
        fig = px.line(
            df_chart, x='Date', y='Valeur', color='Stratégie',
            color_discrete_map={'Notre Stratégie': '#00CC96', 'Stratégie AQR': '#FFA15A', 'S&P 500': '#636EFA'},
            template="plotly_dark", height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- VOLATILITE ---
        st.subheader("Analyse du Risque (Volatilité Roulante)")
        
        # Calcul Volatilité roulante 30 jours
        window = 30
        df_vol = pd.DataFrame({'Date': df_bt['Date']})
        df_vol['Notre Stratégie'] = df_bt['Ret_Our'].rolling(window).std() * np.sqrt(252)
        df_vol['Stratégie AQR'] = df_bt['Ret_AQR'].rolling(window).std() * np.sqrt(252)
        
        df_vol_melt = df_vol.melt('Date', var_name='Stratégie', value_name='Volatilité')
        
        fig_vol = px.line(
            df_vol_melt, x='Date', y='Volatilité', color='Stratégie',
            color_discrete_map={'Notre Stratégie': '#00CC96', 'Stratégie AQR': '#FFA15A'},
            template="plotly_dark", height=350
        )
        fig_vol.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_vol, use_container_width=True)

# --- PAGE 5 : LIVE TERMINAL ---
elif page == "5. Live Strategy":
    st.title("Live Strategy")
    st.caption("Données en temps réel (Base Live)")
    
    df_live = get_live_data()
    
    if df_live.empty:
        st.error("Aucune donnée Live trouvée dans la table 'qmj_scores'.")
    else:
        # Taille du panier fixée par défaut (Filtre supprimé)
        nb_pos = 15
        
        # Sélection Long / Short
        # On suppose que Quality_Score est déjà calculé dans la table
        longs = df_live.nlargest(nb_pos, 'Quality_Score')
        shorts = df_live.nsmallest(nb_pos, 'Quality_Score')
        
        # KPIs Live
        avg_long = longs['Quality_Score'].mean()
        avg_short = shorts['Quality_Score'].mean()
        spread = avg_long - avg_short
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Score Panier Long", f"{avg_long:.2f}")
        c2.metric("Score Panier Short", f"{avg_short:.2f}")
        c3.metric("Quality Spread", f"{spread:.2f}", delta="Alpha Potentiel")
        
        # Graphiques
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Carte des Opportunités")
            # Scatter Plot: Profitability vs Safety
            fig_map = px.scatter(
                df_live, x="z_Profitability", y="z_Safety", 
                color="Quality_Score", size="MarketCap",
                hover_name="Ticker", template="plotly_dark",
                color_continuous_scale="RdYlGn", height=450
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
        with col2:
            st.subheader("Inspecteur d'Action")
            ticker = st.selectbox("Choisir une action", df_live['Ticker'].unique())
            
            # Radar Chart
            row = df_live[df_live['Ticker'] == ticker].iloc[0]
            # On prend les colonnes Z-Scores disponibles
            categories = ['z_Profitability', 'z_Growth', 'z_Safety', 'z_Payout', 'z_Hurst', 'z_Toxic']
            # Vérifions que ces colonnes existent, sinon on met 0
            values = [row.get(c, 0) for c in categories]
            
            fig_r = go.Figure(go.Scatterpolar(
                r=values + values[:1], 
                theta=categories + categories[:1], 
                fill='toself', line_color='#4F8BF9'
            ))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-3, 3])), template="plotly_dark", height=350)
            st.plotly_chart(fig_r, use_container_width=True)
            
        # Tableaux Positions
        c_l, c_s = st.columns(2)

        cols_show = ['Ticker', 'Quality_Score']
        # Sécurité si colonnes manquantes
        cols_show = [c for c in cols_show if c in df_live.columns]
        
        with c_l: 
            st.success("🟢 PANIER LONG (ACHAT)")
            st.dataframe(longs[cols_show], hide_index=True, use_container_width=True)
        with c_s: 
            st.error("🔴 PANIER SHORT (VENTE)")
            st.dataframe(shorts[cols_show], hide_index=True, use_container_width=True)
