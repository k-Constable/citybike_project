#Importing relevant libraries
import pandas as pd
import numpy as np
import pickle as pkl
from datetime import datetime, timedelta
from train import pipeline
def main():
    stations_df = pd.read_csv("stations.csv")
    weather_df = pd.read_csv("weather.csv")
    trips_df = pd.read_csv("trips_size_reduced.csv")

    combined_df = pipeline(stations_df=stations_df, weather_df=weather_df, trips_df=trips_df)

    last_9_rows = combined_df.tail(9)
    X_last_9_rows = last_9_rows.drop(columns=["target_number_of_bikes"])

    model = pkl.load(open('model.pkl', 'rb'))

    predictions = model.predict(X_last_9_rows.drop(columns=["timestamp", "latest_target_measurement"]))
    rounded_predictions = [round(n) for n in predictions]

    print("Siste tidsstempel i data: ", last_9_rows.iloc[8, last_9_rows.columns.get_loc("latest_target_measurement")])
    print("Neste hele klokketime: ", last_9_rows.iloc[8, last_9_rows.columns.get_loc("timestamp")])
    print("Predikerer for tidsstempel: ", last_9_rows.iloc[8, last_9_rows.columns.get_loc("timestamp")] + timedelta(hours=1))

    print('--------------------------------------------------------------------------------------------------')
    print('| Stasjon             | Nåværende Sykler        | Predikerte Sykler       |')
    print('--------------------------------------------------------------------------------------------------')

    for i in range(len(last_9_rows)):
        station_name = "undefined"
        if (last_9_rows.iloc[i, last_9_rows.columns.get_loc("selected_Møllendalsplass")] == 1):
            station_name = "Møllendalsplass      "
        elif (last_9_rows.iloc[i, last_9_rows.columns.get_loc("selected_Torgallmenningen")] == 1):
            station_name = "Torgallmenningen     "
        elif (last_9_rows.iloc[i, last_9_rows.columns.get_loc("selected_Grieghallen")] == 1):
            station_name = "Grieghallen          "
        elif (last_9_rows.iloc[i, last_9_rows.columns.get_loc("selected_Høyteknologisenteret")] == 1):
            station_name = "Høyteknologisenteret "
        elif (last_9_rows.iloc[i, last_9_rows.columns.get_loc("selected_Studentboligene")] == 1):
            station_name = "Studentboligene      "
        elif (last_9_rows.iloc[i, last_9_rows.columns.get_loc("selected_Akvariet")] == 1):
            station_name = "Akvariet             "
        elif (last_9_rows.iloc[i, last_9_rows.columns.get_loc("selected_Damsgårdveien71")] == 1):
            station_name = "Damsgårdsveien 71    "
        elif (last_9_rows.iloc[i, last_9_rows.columns.get_loc("selected_Dreggsallmenningen_Sør")] == 1):
            station_name = "Dreggsallmenningen Sør"
        elif (last_9_rows.iloc[i, last_9_rows.columns.get_loc("selected_Florida")] == 1):
            station_name = "Florida Bybanestopp   "

        current_bikes = last_9_rows.iloc[i, last_9_rows.columns.get_loc("free_bikes")]
        future_bikes = rounded_predictions[i]
        print('  ', station_name, '   ', current_bikes, '                                    ', future_bikes )

if __name__ == "__main__":
    main()