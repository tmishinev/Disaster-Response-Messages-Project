# Disaster Response Messages Project

## Web implementation : https://dr-dash.herokuapp.com/


## 1. About the Project:

    #### This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The initial dataset provided #### by Figure Eight contain real messages sent during disaster events and their respective categories. The aim of the project is to build #### a Natural Language Processing tool that categorize messages.

    #### The Project is divided in the following Sections:

        #### 1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
        #### 2. Machine Learning Pipeline to train a model able to classify text message in categories
        #### 3. Web App to show model results in real time using Dash and Plotly.

## 2. File structure:
- ### **root directory**:
    - train_classifuier.py : trains the classifier (LinearSVC). Run in root directory by - 'python train_classifier.py data/DisasterResponse.db models/classifier.pkl'.
    - dash_app.py : Dash/Plotly web visualization - python dash_app.py.
    - nltk.txt : nltk downloads for Heroku implementation.
    - Procfile : file for Heroku implementation.
    - requirements.txt : required libraries.

- ### **data**:
     - DisasterResponse.db : Database with cleaned data created by 'process_data.py' script.
     - process_data.py : Cleans the raw .csv data and saves it into SQLite database (DisasterResponse.db : table - DisasterMessageETL)

- ### **models**:
     - classifier.pkl : trained classifier created by 'train_classifier.py' script

- ### **custom**:
    - custom_tokens.py : contains custom tokenization function

## 3. Build with:
    
    ### Web app implementation using Dash and Plotly. Hosted by Heroku

## 4. Installation:

    #### - clone the repository.
    #### - run process_data.py to create the SQLlite.db. () : 'python process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
    #### - run train_classifier to train the 'models/classifier.pkl' : 'python train_classifier.py data/DisasterResponse.db models/classifier.pkl'
    #### - run dash_app.py to start the server (localhost:8050). : 'python dash_app.py'

## 5. Web app interface:

- ### Heroku deployment: https://dr-dash.herokuapp.com/


- ### **Tab - Predict message:**

    - Text input control: Enter message for classification

    - fig.1 : Bar chart with predicted categories.

    ![alt text](https://github.com/tmishinev/dr_dash/blob/main/tab.1.JPG?raw=true)

- ### **Tab - Explore Dataset:**

    - Chart 1 : Most common words associated with disaster messages.

    - Chart 2 : Most common words associated with non-disaster messages.

    - Chart 3 : Percentage positive labels per category.

    - Chart 4 : Pie Chart. Message distribution by genre/related to disaster.

    - Slider : Select how many top words to include into Chart 1/2.

    ![alt text](https://github.com/tmishinev/dr_dash/blob/main/tab.2.JPG?raw=true)

- ### 6. Contact:

        Todor Mishinev - todor.mishinev@gmail.com

        Project link - https://github.com/tmishinev/dr_dash.git



