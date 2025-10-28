import pandas as pd
import numpy as np

def calculate_form_features(df, n_matches=5):
 
    df = df.copy()
    
    # Sorter etter sesong og runde
    df = df.sort_values(['sesonger', 'rk'])
    
    # Beregn rullerende gjennomsnitt for viktige metrikker
    for team in df['squad'].unique():
        team_mask = df['squad'] == team
        
        # Scoring form
        df.loc[team_mask, 'avg_goals_scored'] = df.loc[team_mask, 'gf'].rolling(n_matches, min_periods=1).mean()
        df.loc[team_mask, 'avg_goals_conceded'] = df.loc[team_mask, 'ga'].rolling(n_matches, min_periods=1).mean()
        
        # Expected goals form
        df.loc[team_mask, 'avg_xg'] = df.loc[team_mask, 'xg'].rolling(n_matches, min_periods=1).mean()
        df.loc[team_mask, 'avg_xga'] = df.loc[team_mask, 'xga'].rolling(n_matches, min_periods=1).mean()
        
        # Poeng form
        df.loc[team_mask, 'avg_points'] = df.loc[team_mask, 'pts'].rolling(n_matches, min_periods=1).mean()
    
    return df

def calculate_team_stats(df):
    """
    Beregner lagstatistikk for hele sesongen.
    
    Args:
        df: DataFrame med kampdata
    
    Returns:
        DataFrame med nye lag-features
    """
    df = df.copy()
    
    # For hver sesong
    for season in df['sesonger'].unique():
        season_mask = df['sesonger'] == season
        
        # Beregn kumulative statistikker
        for team in df.loc[season_mask, 'squad'].unique():
            team_mask = (df['squad'] == team) & season_mask
            
            # Kumulative mål og xG
            df.loc[team_mask, 'cum_gf'] = df.loc[team_mask, 'gf'].cumsum()
            df.loc[team_mask, 'cum_ga'] = df.loc[team_mask, 'ga'].cumsum()
            df.loc[team_mask, 'cum_xg'] = df.loc[team_mask, 'xg'].cumsum()
            df.loc[team_mask, 'cum_xga'] = df.loc[team_mask, 'xga'].cumsum()
            
            # Kumulativ målforskjell
            df.loc[team_mask, 'cum_gd'] = df.loc[team_mask, 'cum_gf'] - df.loc[team_mask, 'cum_ga']
            df.loc[team_mask, 'cum_xgd'] = df.loc[team_mask, 'cum_xg'] - df.loc[team_mask, 'cum_xga']
    
    return df

def create_match_features(df):
    """
    Lager features for hver kamp basert på begge lags statistikk.
    
    Args:
        df: DataFrame med kampdata
    
    Returns:
        DataFrame med kampspesifikke features
    """
    # Opprett en kopi av dataframet
    match_df = df.copy()
    
    # Legg til form features
    match_df = calculate_form_features(match_df)
    
    # Legg til lagstatistikk
    match_df = calculate_team_stats(match_df)
    
    # Beregn relative styrkeforhold
    match_df['goal_scoring_power'] = match_df['avg_goals_scored'] / match_df['avg_goals_scored'].mean()
    match_df['defense_strength'] = 1 / (match_df['avg_goals_conceded'] / match_df['avg_goals_conceded'].mean())
    
    # Beregn form-indikator (siste 5 kampers poeng / maksimale mulige poeng)
    match_df['form_indicator'] = match_df['avg_points'] / 3
    
    # Fjern NaN-verdier med 0 eller gjennomsnitt
    numeric_columns = match_df.select_dtypes(include=[np.number]).columns
    match_df[numeric_columns] = match_df[numeric_columns].fillna(match_df[numeric_columns].mean())
    
    return match_df

def prepare_features_for_training(df):
    """
    Forbereder features for modelltrening.
    
    Args:
        df: DataFrame med kampdata
    
    Returns:
        X: Feature matrix
        y: Target variabel (seier/tap/uavgjort)
    """
    # Opprett features
    feature_df = create_match_features(df)
    
    # Velg features for trening
    feature_columns = [
        'avg_goals_scored', 'avg_goals_conceded',
        'avg_xg', 'avg_xga',
        'avg_points', 'cum_gd', 'cum_xgd',
        'goal_scoring_power', 'defense_strength',
        'form_indicator'
    ]
    
    X = feature_df[feature_columns]
    
    # Opprett target variabel (1 for seier, 0 for uavgjort, -1 for tap)
    y = np.where(feature_df['pts'] == 3, 1, np.where(feature_df['pts'] == 1, 0, -1))
    
    return X, y
