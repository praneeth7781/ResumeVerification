Again, this folder contains the following data and codes
1. pre-processed data for each user
2. code to convert this processed data to a .csv file, splitting up the data every time there is a newline in the .txt file. This code outputs the CSV file(text, author) to another common folder (0. combined)
3. code to append all CSV files to a common one(CombineCSV.py)
4. Code to train and validate an SVM ML model using the WordFreq feature

For this, right now I have only used the dataset for 4 users, with word count ranging from 67-91k. WIll have to refine this more as this biases the model towards the dataset with 91k words, planning to trim and test the model on 60k words per user. 
Without refinement, have achieved an accuracy of 74%, plan to try other models such as RandomForest and refine the process a bit more to improve accuracy

