Instructions for running the project:

The UI can be started by running main.py. Before doing this, it is important that the database folder(s) are in the root folder of the project. In case there are multiple databases available (we have the raw database, the resampled database, and the normalized database) the database to be shown in the UI can be set in config.py, which can be found in the mesh_browser folder. 
- DATABASE_DIR for the raw database
- REFINED_DATABASE_DIR for the resampled database
- NORMALIZED_DATABASE_DIR for the normalized database

DATABASE_DIR is the only database which meshes can be shown in the UI. If you want to see meshes after resampling or normalization, change the DATABASE_DIR variable to either one of those databases (you can do this simply by commenting out the current "active" DATABASE_DIR and uncommenting whichever one you want to see). Those databases do have to be in the project for this of course.

We recommend having a normalized database in there for the query viewer to work best. Either download it from https://drive.google.com/file/d/1FCpEXWQeYVWjBCnyhyetGPgo53l4e2XB/view?usp=sharing or run the normalization button as explained below.

The feature database is already included in this github repository, so they should not have to be recomputed.

In the config file, the resampling settings can be adjusted, as well as the settings for the K nearest neighbors search.

In the UI you can select subfolders of the database and any of their meshes, and change the display mode.

There are multiple buttons in the UI:
- k-d tree search settings: here you can switch between KNN and RNN, and you can set K and R from here as well.
- Shape analyzer: this button will go through the entire database (set in DATABASE_DIR) and extract data like
- Normalizer: this button normalizes the database
- Descriptors: this button computes the descriptor and puts them in an excel file
- Find similar shapes: the query system from step 4
- Find similar shapes (kdtree): the query system using the kdtree
- Show t-SNE map: Shows the t-SNE map in a separate internet window - it does not compute it, rather show what's been precomputed

To run our code you need to have a few libraries installed. We have compiled them in requirements.txt. You can run this using this command:
pip install -r requirements.txt
