Disaster Response Pipeline Project
Motivation
Created ETL pipeline in Python to clean, prepare and load data to SQLite; Built NLP and ML Model to classify text from messages and posts to 36 disaster labels with MultiOutput Classifier (LinearSVC is used) and deployed working model to WebApp

Project Files
Python Scripting files:
data/process_data.py: Python script to clean, join data in the 2 csv files in the data folder and load data into the sqlite database DisasterResponse.db
models/train_classifier.py: Python script to load data from Database created from process_data.py, train and test RandomForestClassifier Model to categorize messages using NLTK and GridSearchCV
app/run.py: python script to deploy the model onto webapp, where the model outputs the category of each message input in by user
Data files:
data/disaster_categories.csv: contains 36 categories for each message ID
data/disaster_messages.csv: contains actual message text
DisasterResponse.db: sqlite database created from process_data.py
Helper files:
app/go.html, app/master.html: html files to render webapp for run.py output
Installation/Requirement
The code is written in Python3. To run the Python code, you will need to install necessary packages using pip install or conda install.
Heroku files:
Procfile : Necessary for bootstrap/flask deployment in Heroku
Procfile_streamlit : Contains the string for streamlit deployment in Heroku (should replace Procfile)
nltk.txt : corpus to be downloaded during deployment

Data Analysis packages: numpy, pandas
Database engine: sqlalchemy
Natural language processing: NLTK and its dependencies
Machine Learning packages: scikitlearn and its dependencies
Instructions to run the python script
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run deploy web app. python run.py

Go to http://0.0.0.0:3001/ or localhost:3001