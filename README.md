Instructions for running the project:

The UI can be started by running main.py. Before doing this, it is important that the database file(s) are in the root folder of the project. In case there are multiple databases available (we have the raw database, the resampled database, and the normalized database) the database to be shown in the UI can be set in config.py. 
- DATABASE_DIR for the raw database
- REFINED_DATABASE_DIR for the resampled database
- NORMALIZED_DATABASE_DIR for the normalized database

DATABASE_DIR is the only database which meshes can be shown in the UI. If you want to see meshes after resampling or normalization, change the DATABASE_DIR variable to either one of those databases (you can do this simply by commenting out the current "active" DATABASE_DIR and uncommenting whichever one you want to see.

In this config file, the resampling settings can be adjusted, as well as the settings for the K nearest neighbors search.

In the UI you can select subfolders of the database and any of their meshes, and change the display mode.

There are multiple buttons in the UI:
k-d tree search settings: here you can switch between KNN and RNN, and you can set K and R from here as well.
Shape analyzer: this button will go through the entire database (set in DATABASE_DIR) and extract data like 
