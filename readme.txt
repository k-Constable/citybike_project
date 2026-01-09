The goal of this project was to predict the number of available Bergen City Bikes at 9 selected stations an hour ahead of time. 

The stations are:
  - Møllendalsplass 
  - Torgallmenningen
  - Grieghallen
  - Høyteknologisenteret
  - Studentboligene
  - Akvariet
  - Damsgårdveien 71
  - Dreggsallmenningen Sør 
  - Florida Bybanestopp


To predict the number of free bikes for the next hour, run the train.py file (python train.py) in the terminal,
then run the predict file (python predict.py) in the terminal. This should print a readout in the terminal, showing the number of bikes at the last measured point in the dataset, as well as predictions for the following hour. 

For comprehensive coverage of the project, see the project_report pdf. 

Additional note:
Because of the limits to file sizes in GitHub, the dataset containing 'trips' (trips_size_reduced.csv) had to be largely reduced in size. Though the scripts work the same, it is worth mentioning as the accuracy of the predictions may be reduced. The report was made using the entire 'trips' dataset, and the results may differ slightly.




