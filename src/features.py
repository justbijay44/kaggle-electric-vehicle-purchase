from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC_COLS = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
                'Environmental_Concern_Level']

NOMINAL_COLS = ['Gender', 'Current_Car_Type']

BINARY_COLS = ['Home_Charging_Possible', 'Subsidy_Available']

ORDINAL_COLS = ['City_Type', 'Range_Anxiety_Level']

def build_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_COLS),
        ('nom', OneHotEncoder(handle_unknown='ignore'), NOMINAL_COLS),
        ('bin', OrdinalEncoder(categories=[['No', 'Yes'], ['No', 'Yes']]), BINARY_COLS),
        ('ord', OrdinalEncoder(categories=[['Rural', 'Suburban', 'Urban'], ['Low', 'Medium', 'High']]), ORDINAL_COLS),
    ])

def compute_combo_map(train_feat, train_target):
    combo_fit = train_feat['Subsidy_Available'] + "_" + train_feat['Range_Anxiety_Level']
    return train_target.groupby(combo_fit).mean()

def apply_combo_map(feat_to_encode, combo_map):
    out = feat_to_encode.copy()
    combo_apply = feat_to_encode['Subsidy_Available'] + "_" + feat_to_encode['Range_Anxiety_Level']
    out['Combo_te'] = combo_apply.map(combo_map)
    return out