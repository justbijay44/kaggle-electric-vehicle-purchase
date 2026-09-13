import pandas as pd
from src.features import build_preprocessor, compute_combo_map, apply_combo_map

def test_build_preprocessor_output_shape():
    df = pd.DataFrame({
        'Age': [25, 40], 'Annual_Income_USD': [50000, 80000],
        'Daily_Commute_km': [10, 20], 'Number_of_Cars_Owned': [1, 2],
        'Charging_Stations_Near_Home': [2, 5], 'Charging_Stations_Near_Work': [3, 6],
        'Environmental_Concern_Level': [2, 4],
        'Gender': ['Male', 'Female'], 'Current_Car_Type': ['Sedan', 'SUV'],
        'Home_Charging_Possible': ['Yes', 'No'], 'Subsidy_Available': ['No', 'Yes'],
        'City_Type': ['Urban', 'Rural'], 'Range_Anxiety_Level': ['Low', 'High'],
    })

    preprocessor = build_preprocessor()
    result = preprocessor.fit_transform(df)
    assert result.shape[0] == 2

def test_combo_map_encoding():
    X = pd.DataFrame({
        'Subsidy_Available': ['Yes', 'Yes', 'No'],
        'Range_Anxiety_Level': ['Low', 'Low', 'High'],
    })

    y = pd.Series([1, 0, 0])

    combo_map = compute_combo_map(X, y)
    result = apply_combo_map(X, combo_map)

    assert result['Combo_te'].iloc[0] == 0.5
    assert result['Combo_te'].iloc[2] == 0