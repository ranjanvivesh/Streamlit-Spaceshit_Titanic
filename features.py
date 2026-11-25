import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
BINARY = ['CryoSleep', 'VIP']
CATEGORICAL = ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide']
FEATURES = NUMERIC + BINARY + CATEGORICAL


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cabin = df['Cabin'].str.split('/', expand=True).reindex(columns=[0, 1, 2])
    df['CabinDeck'] = cabin[0]
    df['CabinSide'] = cabin[2]

    for col in BINARY:
        df[col] = df[col].map({True: 1, False: 0})

    return df[FEATURES]


def make_preprocessor() -> ColumnTransformer:
    numeric = SimpleImputer(strategy='median')
    binary = SimpleImputer(strategy='most_frequent')
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    return ColumnTransformer([
        ('numeric', numeric, NUMERIC),
        ('binary', binary, BINARY),
        ('categorical', categorical, CATEGORICAL),
    ])