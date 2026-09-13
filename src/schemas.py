from pydantic import BaseModel

class EVFeatures(BaseModel):
    Age                         : int
    Annual_Income_USD           : float
    Daily_Commute_km            : float
    Number_of_Cars_Owned        : int
    Charging_Stations_Near_Home : int
    Charging_Stations_Near_Work : int
    Environmental_Concern_Level : float
    Gender                      : str
    City_Type                   : str
    Current_Car_Type            : str
    Home_Charging_Possible      : str
    Subsidy_Available           : str
    Range_Anxiety_Level         : str