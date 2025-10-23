"""Simple training script for Serie A match outcome classifier.

Usage:
  python train.py --data path/to/matches.csv
If no data file is provided, a small synthetic dataset will be generated for demonstration.

The script trains a multinomial logistic regression model inside a scikit-learn Pipeline
that includes simple preprocessing, and saves the fitted pipeline to models/model.joblib.
"""
import argparse
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score


def make_synthetic(n_matches=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    teams = ['Atalanta','Bologna','Cremonese','Empoli','Fiorentina','Inter','Juventus','Lazio','Milan','Monza','Napoli','Roma','Salernitana','Sassuolo','Spezia','Torino','Udinese','Verona']
    rows = []
    date_range = pd.date_range('2022-08-01', periods=n_matches, freq='D')
    for i in range(n_matches):
        home = rng.choice(teams)
        away = rng.choice([t for t in teams if t != home])
        # simple latent strengths
        home_strength = 5 + (len(home) % 5) + rng.normal(0,1)
        away_strength = 5 + (len(away) % 5) + rng.normal(0,1)
        home_adv = 1.0
        home_goals = max(0, int(np.round(np.clip(rng.normal(home_strength/2 + home_adv, 1.4),0,8))))
        away_goals = max(0, int(np.round(np.clip(rng.normal(away_strength/2, 1.4),0,8))))
        rows.append({
            'date': date_range[i],
            'homeTeam': home,
            'awayTeam': away,
            'homeGoals': home_goals,
            'awayGoals': away_goals,
        })
    return pd.DataFrame(rows)


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Build simple features from match dataframe rows.

    Expects columns: homeTeam, awayTeam, homeGoals, awayGoals, (optional) date
    Produces features: homeTeam, awayTeam, homeAdv (1), recent form placeholders.
    """
    def __init__(self):
        self.team_index_ = {}

    def fit(self, X, y=None):
        teams = pd.unique(pd.concat([X['homeTeam'], X['awayTeam']]))
        self.team_index_ = {t:i for i,t in enumerate(sorted(teams))}
        return self

    def transform(self, X):
        # Create simple features
        df = X.copy()
        df = df.reset_index(drop=True)
        # Basic features
        out = pd.DataFrame()
        out['homeTeam'] = df['homeTeam']
        out['awayTeam'] = df['awayTeam']
        out['homeAdv'] = 1.0
        # outcome (target) must be created earlier
        return out


def prepare_target(df):
    # 3-class target from goals
    def label(row):
        if row['homeGoals'] > row['awayGoals']:
            return 'home'
        elif row['homeGoals'] < row['awayGoals']:
            return 'away'
        else:
            return 'draw'
    return df.apply(label, axis=1)


def build_pipeline():
    # categorical team features encoded via OneHot (simple baseline)
    cat_cols = ['homeTeam','awayTeam']
    preproc = ColumnTransformer([
        ('teams', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), ['homeAdv']),
    ])

    pipe = Pipeline([
        ('pre', preproc),
        ('clf', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=2000))
    ])
    return pipe


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, help='Path to matches CSV file')
    p.add_argument('--out', type=str, default='models/model.joblib', help='Path to save trained model')
    args = p.parse_args(argv)

    if args.data:
        if not os.path.exists(args.data):
            raise SystemExit(f"Data file not found: {args.data}")
        # read a tiny sample to inspect columns
        sample = pd.read_csv(args.data, nrows=5)
        has_date = 'date' in sample.columns
        df = pd.read_csv(args.data, parse_dates=['date'] if has_date else None)
        # If the CSV is season/team-level (has squad & pts), synthesize simple match rows
        if not {'homeTeam','awayTeam','homeGoals','awayGoals'}.issubset(df.columns):
            if {'squad','pts'}.issubset(df.columns):
                print('Detected season/team-level CSV — creating a simple synthetic match dataset from team strengths (very basic)')
                def synth_from_seasons(season_df, matches_per_season=300, random_state=42):
                    rng = np.random.default_rng(random_state)
                    rows = []
                    seasons = season_df['sesonger'].unique() if 'sesonger' in season_df.columns else [None]
                    for s in seasons:
                        sub = season_df[season_df['sesonger']==s] if s is not None else season_df
                        teams = sub['squad'].tolist()
                        if len(teams) < 2:
                            continue
                        pts_map = dict(zip(sub['squad'], sub['pts']))
                        max_pts = max(pts_map.values()) if pts_map else 1
                        for i in range(matches_per_season):
                            h = rng.choice(teams)
                            a = rng.choice([t for t in teams if t != h])
                            # simple strength proportional to pts
                            hs = (pts_map.get(h, 0) / max_pts) * 2.5 + rng.normal(0,0.3)
                            as_ = (pts_map.get(a, 0) / max_pts) * 2.5 + rng.normal(0,0.3)
                            hs += 0.3  
                            hg = int(max(0, rng.poisson(max(0.2, hs))))
                            ag = int(max(0, rng.poisson(max(0.2, as_))))
                            rows.append({'date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i), 'homeTeam': h, 'awayTeam': a, 'homeGoals': hg, 'awayGoals': ag})
                    return pd.DataFrame(rows)

                df = synth_from_seasons(df)
            else:
                raise SystemExit('CSV does not contain match-level columns and is not recognized as a season/team table (needs squad & pts)')
    else:
        print('No data file provided — generating synthetic data for demo')
        df = make_synthetic(1000)


    required_cols = {'homeTeam','awayTeam','homeGoals','awayGoals'}
    if not required_cols.issubset(df.columns):
        raise SystemExit(f'Missing required columns. Found: {df.columns.tolist()}')

    y = prepare_target(df)

    X = df[['homeTeam','awayTeam']].copy()
    X['homeAdv'] = 1.0

    if 'date' in df.columns:
        df_sorted = df.sort_values('date')
        split_idx = int(len(df_sorted)*0.8)
        train_idx = df_sorted.index[:split_idx]
        test_idx = df_sorted.index[split_idx:]
        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_pipeline()
    print('Training model...')
    pipe.fit(X_train, y_train)

    print('Evaluating...')
    y_pred_proba = pipe.predict_proba(X_test)
    classes = pipe.classes_
    ll = log_loss(y_test, y_pred_proba, labels=classes)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f'Log-loss: {ll:.4f}, Accuracy: {acc:.3f}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    print(f'Model saved to {out_path}')


if __name__ == '__main__':
    main()
