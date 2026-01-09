#Importing relevant libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import pickle as pkl

def pipeline(stations_df, weather_df, trips_df):
    #Sorterer stations datasettet etter tidspunkt og stasjon
    stations_df_sorted = stations_df.sort_values(["station", "timestamp"]).reset_index()


    #Henter ut indexen til siste måling i hver time
    latest_hour_index = []
    for i in range(len(stations_df_sorted)-1):
        if stations_df_sorted["timestamp"][i][11:13] != stations_df_sorted["timestamp"][i+1][11:13]:
            latest_hour_index.append(i)


    #Lager en ny df med kun de siste målingene per time i stations_df, og gjør alle timestampsene om til datetime verdier. Så lager jeg en ny kolonne med opprundede verdier
    latest_hour_station_df = stations_df_sorted.iloc[latest_hour_index].copy().reset_index()
    latest_hour_station_df["timestamp"] = pd.to_datetime(latest_hour_station_df["timestamp"])
    latest_hour_station_df["timestamp_rounded"] = latest_hour_station_df["timestamp"].dt.ceil('h')
    #Dropper de ekstra kolonnene som ble laget
    latest_hour_station_df = latest_hour_station_df.drop(["level_0", "index"], axis=1)


    #Lager en periode med timestamps for å fylle inn timer som ikke har målinger
    latest_hour_station_df.sort_values(["timestamp_rounded", "station"], inplace=True)
    first_timestamp = latest_hour_station_df.iloc[0, -1]
    last_timestamp = latest_hour_station_df.iloc[-1, -1]
    full_range = pd.date_range(first_timestamp, last_timestamp, freq='h', tz='UTC')


    #Omformaterer stations_df slik at indekseringen er tidspunktet, kolonnene er stasjoner og verdiene i kolonnene er antall ledige sykler på de stasjonene
    reformated_station_df = latest_hour_station_df.pivot(index="timestamp_rounded", columns="station", values="free_bikes")


    #Gjør samme pivot operasjon med siste målingstitdspunktet, slik at jeg kan ha riktig tidspunkt sammen med den opprundede verdien
    #Lager en ny kolonne med siste måling per time for de relevante stasjonene vi skal predikere for, og dropper de andre
    measurement_pivot_table = latest_hour_station_df.pivot(index="timestamp_rounded", columns="station", values="timestamp")
    measurement_pivot_table = measurement_pivot_table[["Møllendalsplass", "Torgallmenningen", "Grieghallen", "Høyteknologisenteret", "Studentboligene", "Akvariet", "Damsgårdsveien 71", "Dreggsallmenningen Sør", "Florida Bybanestopp"]]
    measurement_pivot_table["latest_target_measurement"] = measurement_pivot_table.max(axis=1)
    measurement_pivot_table.drop(["Møllendalsplass", "Torgallmenningen", "Grieghallen", "Høyteknologisenteret", "Studentboligene", "Akvariet", "Damsgårdsveien 71", "Dreggsallmenningen Sør", "Florida Bybanestopp"], axis=1, inplace=True)


    #Setter sammen siste-tidspunkt- og stasjons-dataframene
    reformated_station_df = pd.concat([reformated_station_df, measurement_pivot_table], axis=1)


    #Bruker datointervallet som matcher datesettet, og oppretter et datasett med en rad per time for alle timene i det tidsintervallet 
    #Timer uten målinger tar verdiene sine fra forrige måling (forward fill), pga. Last Observation Carried Forward prinsippet
    reformated_station_df = reformated_station_df.reindex(full_range).ffill().rename_axis("timestamp_rounded").reset_index().rename_axis(columns={'station':'index'})


    #Kan fjerne stasjonene vi ikke skal predikere for, hvis treningen tar for lang tid med alle
    reformated_station_df = reformated_station_df[["timestamp_rounded", "Møllendalsplass", "Torgallmenningen", "Grieghallen", "Høyteknologisenteret", "Studentboligene", "Akvariet", "Damsgårdsveien 71", "Dreggsallmenningen Sør", "Florida Bybanestopp", "latest_target_measurement"]]


    #Gjør timestamp i weather om til datetime, og gir kolonnen i reformated_station_df samme navn, slik at de kan merges
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], utc=True)
    reformated_station_df.rename(columns={"timestamp_rounded":"timestamp"}, inplace=True)

    #Kombinerer stasjon og værdatasett på felles tidspunkt
    stations_weather_df = reformated_station_df.merge(weather_df, on="timestamp")


    #Fjerner uønskede rader fra trips, og gjør starttid om til datetime 
    trips_df_floored = trips_df[["start_station_name", "started_at"]]
    trips_df_floored["started_at"] = pd.to_datetime(trips_df_floored["started_at"], format="mixed").dt.ceil('h')


    #Dropper startstasjonsnavn
    trips_df_floored = trips_df_floored.drop("start_station_name", axis=1)

    #Omformaterer til å vise totale turer på hver time
    trips_per_hour = pd.DataFrame(trips_df_floored.groupby(["started_at"]).value_counts()).reset_index()

    #Gir nye navn for å reflektere de nye verdiene
    trips_per_hour.rename(columns={"started_at":"timestamp", "count":"traffic"}, inplace=True)

    #Utvider trip datasettet til å inneholde alle timer, fyller timer uten målinger med 0
    trips_per_hour_filled = trips_per_hour.set_index("timestamp")
    trips_per_hour_filled = trips_per_hour_filled.reindex(full_range).reset_index().rename(columns={"index":"timestamp"}).fillna(0)

    #Normaliserer totale turer per time for å gi en generell måling for hvor mye aktivitet "traffic" det er med bysyklene for en tilhørende time
    #scaler = MinMaxScaler()
    #trips_per_hour_filled["traffic_coefficient"] = scaler.fit_transform(trips_per_hour_filled[["traffic_coefficient"]])

    #Setter trips datasettet sammen med weather og stations
    combined_df = stations_weather_df.merge(trips_per_hour_filled, on="timestamp")


    #Legger til rader som skal vise hvilken stasjon som er valgt, og en rad til target verdier
    #Legger til en rad free_bikes som skal vise antall ledige sykler på den stasjonen som er valgt, slik at man ikke trenger en kolonne per stasjon
    expanded_df = combined_df.copy()
    expanded_df["free_bikes"] = 0
    expanded_df["selected_Møllendalsplass"] = 0
    expanded_df["selected_Torgallmenningen"] = 0
    expanded_df["selected_Grieghallen"] = 0
    expanded_df["selected_Høyteknologisenteret"] = 0
    expanded_df["selected_Studentboligene"] = 0
    expanded_df["selected_Akvariet"] = 0
    expanded_df["selected_Damsgårdveien71"] = 0
    expanded_df["selected_Dreggsallmenningen_Sør"] = 0
    expanded_df["selected_Florida"] = 0
    expanded_df["target_number_of_bikes"] = np.nan

    #Kopierer hver rad 9 ganger 
    expanded_9_df = pd.concat([expanded_df] * 9, ignore_index=True).sort_values("timestamp").reset_index().drop("index", axis=1)
    pd.set_option("display.max_columns", None)

    #Gjør at en stasjon er valgt per rad, og setter target verdien til den valgte stasjonens antall sykler om en time
    j = 0
    for i in range(expanded_9_df.shape[0]-9):
        if j == 0:
            expanded_9_df.iloc[i, 16] = 1
            expanded_9_df.iloc[i, 15] = expanded_9_df.iloc[i, 1]
            expanded_9_df.iloc[i, 25] = expanded_9_df.iloc[i+9, 1]
            j += 1
        elif j == 1:
            expanded_9_df.iloc[i, 17] = 1
            expanded_9_df.iloc[i, 15] = expanded_9_df.iloc[i, 2]
            expanded_9_df.iloc[i, 25] = expanded_9_df.iloc[i+9, 2]
            j += 1
        elif j == 2:
            expanded_9_df.iloc[i, 18] = 1
            expanded_9_df.iloc[i, 15] = expanded_9_df.iloc[i, 3]
            expanded_9_df.iloc[i, 25] = expanded_9_df.iloc[i+9, 3]
            j += 1
        elif j == 3:
            expanded_9_df.iloc[i, 19] = 1
            expanded_9_df.iloc[i, 15] = expanded_9_df.iloc[i, 4]
            expanded_9_df.iloc[i, 25] = expanded_9_df.iloc[i+9, 4]
            j += 1
        elif j == 4:
            expanded_9_df.iloc[i, 20] = 1
            expanded_9_df.iloc[i, 15] = expanded_9_df.iloc[i, 5]
            expanded_9_df.iloc[i, 25] = expanded_9_df.iloc[i+9, 5]
            j += 1
        elif j == 5:
            expanded_9_df.iloc[i, 21] = 1
            expanded_9_df.iloc[i, 15] = expanded_9_df.iloc[i, 6]
            expanded_9_df.iloc[i, 25] = expanded_9_df.iloc[i+9, 6]
            j += 1
        elif j == 6:
            expanded_9_df.iloc[i, 22] = 1
            expanded_9_df.iloc[i, 15] = expanded_9_df.iloc[i, 7]
            expanded_9_df.iloc[i, 25] = expanded_9_df.iloc[i+9, 7]
            j += 1
        elif j == 7:
            expanded_9_df.iloc[i, 23] = 1
            expanded_9_df.iloc[i, 15] = expanded_9_df.iloc[i, 8]
            expanded_9_df.iloc[i, 25] = expanded_9_df.iloc[i+9, 8]
            j += 1
        elif j == 8:
            expanded_9_df.iloc[i, 24] = 1
            expanded_9_df.iloc[i, 15] = expanded_9_df.iloc[i, 9]
            expanded_9_df.iloc[i, 25] = expanded_9_df.iloc[i+9, 9]
            j = 0
    #Fyller de siste 9 radene manuelt, siden siste kolonne ikke har noen verdi å ta fra, og gjør de til 0 slik at de ikke gir en feilmelding
    expanded_9_df.iloc[-9, 16] = 1
    expanded_9_df.iloc[-9, 15] = expanded_9_df.iloc[-9, 1]
    expanded_9_df.iloc[-9, 25] = 0

    expanded_9_df.iloc[-8, 17] = 1
    expanded_9_df.iloc[-8, 15] = expanded_9_df.iloc[-8, 2]
    expanded_9_df.iloc[-8, 25] = 0

    expanded_9_df.iloc[-7, 18] = 1
    expanded_9_df.iloc[-7, 15] = expanded_9_df.iloc[-7, 3]
    expanded_9_df.iloc[-7, 25] = 0

    expanded_9_df.iloc[-6, 19] = 1
    expanded_9_df.iloc[-6, 15] = expanded_9_df.iloc[-6, 4]
    expanded_9_df.iloc[-6, 25] = 0

    expanded_9_df.iloc[-5, 20] = 1
    expanded_9_df.iloc[-5, 15] = expanded_9_df.iloc[-5, 5]
    expanded_9_df.iloc[-5, 25] = 0

    expanded_9_df.iloc[-4, 21] = 1
    expanded_9_df.iloc[-4, 15] = expanded_9_df.iloc[-4, 6]
    expanded_9_df.iloc[-4, 25] = 0

    expanded_9_df.iloc[-3, 22] = 1
    expanded_9_df.iloc[-3, 15] = expanded_9_df.iloc[-3, 7]
    expanded_9_df.iloc[-3, 25] = 0

    expanded_9_df.iloc[-2, 23] = 1
    expanded_9_df.iloc[-2, 15] = expanded_9_df.iloc[-2, 8]
    expanded_9_df.iloc[-2, 25] = 0

    expanded_9_df.iloc[-1, 24] = 1
    expanded_9_df.iloc[-1, 15] = expanded_9_df.iloc[-1, 9]
    expanded_9_df.iloc[-1, 25] = 0

    #Fjerner kolonnene per stasjon, siden jeg nå har en felles
    expanded_9_df.drop(columns=["Møllendalsplass", "Torgallmenningen", "Grieghallen", "Høyteknologisenteret", "Studentboligene", "Akvariet", "Damsgårdsveien 71", "Dreggsallmenningen Sør", "Florida Bybanestopp"], inplace=True)


    #Konverterer alle datetime verdiene til riktig tidssone
    expanded_9_df["timestamp"] = expanded_9_df["timestamp"].dt.tz_convert("cet")
    expanded_9_df["latest_target_measurement"] = expanded_9_df["latest_target_measurement"].dt.tz_convert("cet")


    #Deler opp timestamp i måneder, dager og timer siden dataen skal brukes på regresjonsmodeller, og de ikke tar Datetime objekter (siden det er som en string)
    expanded_9_df["hour"] = expanded_9_df["timestamp"].dt.hour
    expanded_9_df["day_of_week"] = expanded_9_df["timestamp"].dt.day_of_week
    expanded_9_df["month"] = expanded_9_df["timestamp"].dt.month


    expanded_9_df.head(15)

    #Flytter target kolonnen til siste kolonne i dataframen, for ordens skyld.
    target_column = expanded_9_df.pop("target_number_of_bikes")
    expanded_9_df["target_number_of_bikes"] = target_column

    #Forward filler manglende data, og fjerner rader som fortsatt mangler verdier (typisk de første radene som ikke har verdier å forward fille fra)
    expanded_9_df = expanded_9_df.ffill()
    expanded_9_df = expanded_9_df.dropna()

    #Lagrer endelig datasett
    model_ready = expanded_9_df.copy()

    #Eksporterer til .csv fil
    #model_ready.to_csv('model_ready.csv', index=False)
    return model_ready

def model_selection(X_train, y_train, X_val, y_val, X_train_normalised, X_val_normalised):
    # Baseline model
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    y_pred_train_baseline = baseline.predict(X_train)
    y_pred_val_baseline = baseline.predict(X_val)
    train_rmse_baseline = root_mean_squared_error(y_true=y_train, y_pred=y_pred_train_baseline)
    val_rmse_baseline = root_mean_squared_error(y_true=y_val, y_pred=y_pred_val_baseline)

    #Trying different models

    #Random forest regressor
    n_estimators = np.arange(100, 1000, 100)
    rf_models = {n: RandomForestRegressor(n_estimators=n, random_state=42) for n in n_estimators}

    train_rmses_rf = {}
    val_rmses_rf = {}

    for n, model in rf_models.items():
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        train_rmses_rf[n] = root_mean_squared_error(y_true=y_train, y_pred=y_pred_train)
        val_rmses_rf[n] = root_mean_squared_error(y_true=y_val, y_pred=y_pred_val)

    rmses_rf_df = pd.DataFrame.from_dict(train_rmses_rf, orient='index', columns=["train_rmse"])
    rmses_rf_df["val_rmse"] = val_rmses_rf.values()
    sorted_rmses_rf_df = rmses_rf_df.copy()
    sorted_rmses_rf_df.sort_values(by="val_rmse", inplace=True)

    #Choosing best Random Forest model
    best_n_estimators_rf = sorted_rmses_rf_df.index[0]
    best_rmse_rf = rmses_rf_df.loc[best_n_estimators_rf, "val_rmse"]
    best_rf_model = rf_models[best_n_estimators_rf]

    #Support Vector Machine (Support Vector Regressor)
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    svr_models = {kernel: SVR(kernel=kernel) for kernel in kernels}

    train_rmses_svr = {}
    val_rmses_svr = {}

    for kernel, model in svr_models.items():
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        train_rmses_svr[kernel] = root_mean_squared_error(y_true=y_train, y_pred=y_pred_train)
        val_rmses_svr[kernel] = root_mean_squared_error(y_true=y_val, y_pred=y_pred_val)

    rmses_svr_df = pd.DataFrame.from_dict(train_rmses_svr, orient="index", columns=["train_rmse"])
    rmses_svr_df["val_rmse"] = val_rmses_svr.values()
    sorted_rmses_svr_df = rmses_svr_df.copy()
    sorted_rmses_svr_df.sort_values(by="val_rmse", inplace=True)

    #Choosing best Support Vector Machine Model
    best_kernel_svr = sorted_rmses_svr_df.index[0]
    best_rmse_svr = rmses_svr_df.loc[best_kernel_svr, "val_rmse"]
    best_svr_model = svr_models[best_kernel_svr]

    #Multinomial Naive Bayes
    alphas = np.arange(0.01, 10, 1)
    mnb_models = {alpha: MultinomialNB(alpha=alpha) for alpha in alphas}

    train_rmses_mnb = {}
    val_rmses_mnb = {}

    for alpha, model in mnb_models.items():
        model.fit(X_train_normalised, y_train)
        
        y_pred_train = model.predict(X_train_normalised)
        y_pred_val = model.predict(X_val_normalised)

        train_rmses_mnb[alpha] = root_mean_squared_error(y_true=y_train, y_pred=y_pred_train)
        val_rmses_mnb[alpha] = root_mean_squared_error(y_true=y_val, y_pred=y_pred_val)

    rmses_mnb_df = pd.DataFrame.from_dict(train_rmses_mnb, orient='index', columns=["train_rmse"])
    rmses_mnb_df["val_rmse"] = val_rmses_mnb.values()

    sorted_rmses_mnb_df = rmses_mnb_df.copy()
    sorted_rmses_mnb_df.sort_values(by="val_rmse", inplace=True)

    #Choosing best Multinomial Naive Bayes model
    best_alpha_mnb = sorted_rmses_mnb_df.index[0]
    best_rmse_mnb = sorted_rmses_mnb_df.loc[best_alpha_mnb, "val_rmse"]
    best_mnb_model = mnb_models[best_alpha_mnb]


    #Ridge regression model
    alphas = np.arange(0.01, 10, 1)
    rr_models = {alpha: Ridge(alpha=alpha) for alpha in alphas}

    train_rmses_rr = {}
    val_rmses_rr = {}

    for alpha, model in rr_models.items():
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        train_rmses_rr[alpha] = root_mean_squared_error(y_true=y_train, y_pred=y_pred_train)
        val_rmses_rr[alpha] = root_mean_squared_error(y_true=y_val, y_pred=y_pred_val)

    rmses_rr_df = pd.DataFrame.from_dict(train_rmses_rr, orient='index', columns=["train_rmse"])
    rmses_rr_df["val_rmse"] = val_rmses_rr.values()

    sorted_rmses_rr_df = rmses_rr_df.copy()
    sorted_rmses_rr_df.sort_values(by="val_rmse", inplace=True)

    #Choosing best ridge regression model
    best_alpha_rr = sorted_rmses_rr_df.index[0]
    best_rmse_rr = sorted_rmses_rr_df.loc[best_alpha_rr, "val_rmse"]
    best_rr_model = rr_models[best_alpha_mnb]

    # Comparison of models
    comparison_df = pd.DataFrame({
        "Name" : ["Baseline", "Random Forest Regressor", "Support Vector Machine (Regressor)", "Multinomial Naive Bayes", "Ridge Regressor"],
        "RMSES" : [val_rmse_baseline, best_rmse_rf, best_rmse_svr, best_rmse_mnb, best_rmse_rr],
        "Models" : [baseline, best_rf_model, best_svr_model, best_mnb_model, best_rr_model]
        
    }
    )
    #comparison_df = comparison_df.rename({0:"Baseline", 1:"Random Forest Regressor", 2:"Support Vector Machine (Regressor)", 3:"Multinomial Naive Bayes", 4:"Ridge Regressor"})
    comparison_df.sort_values(by="RMSES", inplace=True)
    best_model = {"Name": comparison_df.iloc[0, 0],
                "RMSE": comparison_df.iloc[0, 1],
                "Model": comparison_df.iloc[0, 2]}
    return best_model

def evaluate(best_model, X_test, y_test, X_test_normalised):
    best_model_name = best_model["Name"]
    best_model_val_rmse = best_model["RMSE"]
    best_model_model = best_model["Model"]
    if(best_model_name == "Multinomial Naive Bayes"):
        predictions = best_model_model.predict(X_test_normalised)
    else:
        predictions = best_model_model.predict(X_test)
    test_rmse = root_mean_squared_error(y_pred=predictions, y_true=y_test)
    print('-----------------------------------------------------------------------------------------------------------------------------------')
    print('| Expected RMSE: ', test_rmse, ' | Validation RMSE: ', best_model_val_rmse, ' | Best Model: ', best_model_name, '|')
    print('-----------------------------------------------------------------------------------------------------------------------------------')

def main():
    #Importing raw data
    stations_df = pd.read_csv("stations.csv")
    weather_df = pd.read_csv("weather.csv")
    trips_df = pd.read_csv("trips.csv")

    #Removing the large gap from train - reasoning for this will be provided in the report
    stations_df.drop(stations_df[stations_df["timestamp"] < "2024-09-10 00:00+00:00"].index, inplace=True)

    #Train test split the data
    #70% training data, 30% total val and test
    stations_train, stations_rest = train_test_split(stations_df, train_size=0.7, shuffle=False)
    stations_val, stations_test = train_test_split(stations_rest, train_size=0.5, shuffle=False)

    #weather_train, weather_rest = train_test_split(weather_df, train_size=0.7, shuffle=False)
    #weather_val, weather_test = train_test_split(weather_rest, train_size=0.5, shuffle=False)

    #trips_train, trips_rest = train_test_split(trips_df, train_size=0.7, shuffle=False)
    #trips_val, trips_test = train_test_split(trips_rest, train_size=0.5, shuffle=False)


    #Removing the large gap from train - reasoning for this will be provided in the report
    #stations_train.drop(stations_train[stations_train["timestamp"] < "2024-09-10 00:00+00:00"].index, inplace=True)

    #Running the data through the pipelines
    train = pipeline(stations_df=stations_train, weather_df=weather_df, trips_df=trips_df)
    val = pipeline(stations_df=stations_val, weather_df=weather_df, trips_df=trips_df)
    test = pipeline(stations_df=stations_test, weather_df=weather_df, trips_df=trips_df)


    #Dropping non numerical columns because they don't work with regression
    train.drop(columns=["timestamp", "latest_target_measurement"], inplace=True)
    val.drop(columns=["timestamp", "latest_target_measurement"], inplace=True)
    test.drop(columns=["timestamp", "latest_target_measurement"], inplace=True)




    # Splitting data into features and target variable
    X_train = train.drop(columns=["target_number_of_bikes"])
    y_train = train.target_number_of_bikes.values

    X_val = val.drop(columns=["target_number_of_bikes"])
    y_val = val.target_number_of_bikes.values

    X_test = test.drop(columns=["target_number_of_bikes"])
    y_test = test.target_number_of_bikes.values


    # Because some models can't take negative values, and others perform better on scaled data, I will create a normalised version of our datasets.
    mmscaler = MinMaxScaler()
    X_train_normalised =pd.DataFrame(mmscaler.fit_transform(X=X_train), columns=X_train.columns.values.tolist())
    X_val_normalised = pd.DataFrame(mmscaler.fit_transform(X=X_val), columns=X_train.columns.values.tolist())
    X_test_normalised = pd.DataFrame(mmscaler.fit_transform(X=X_test), columns=X_train.columns.values.tolist())

    #Finding the best model
    best_model = model_selection(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_train_normalised=X_train_normalised, X_val_normalised=X_val_normalised)

    #Evaluating 
    evaluate(best_model=best_model, X_test=X_test, y_test=y_test, X_test_normalised=X_test_normalised)

    #Saving model
    best_model_model = best_model["Model"]
    pkl.dump(best_model_model, open('model.pkl', 'wb'))

if __name__ == "__main__":
    main()