"""
Data: dummy Data
Technique: Supervised, Classification
Algorithm: Naive Bayes Classifier
"""

#creating dummy dataset with 3 variables
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

#label encoding - required to convert string labels of our data into numbers
from sklearn import preprocessing
#initializing an object for LabelEncoder class which helps to convert n string labels for a variable into 0 to n-1 number variables
le_obj = preprocessing.LabelEncoder()
weather_encoded = le_obj.fit_transform(weather)
print("Before encoding the variable weather:",weather)
print("After encoding the variable weather:",weather_encoded)
print("\n")
temp_encoded = le_obj.fit_transform(temp)
print("Before encoding the variable tempreture:",temp)
print("After encoding the variable tempreture:",temp_encoded)
print("\n")
play_encoded = le_obj.fit_transform(play)
print("Before encoding the variable play:",play)
print("After encoding the variable play:",play_encoded)
print("\n")

#creating X (Predictor Variables) in the form of list of tuples
predictor_variables = list(zip(weather_encoded, temp_encoded))
print("The predictor variables are:",predictor_variables)
#creating Y (Target Variable) 
target_variable = play_encoded
print("The target variables are:",target_variable)

#fitting naive bayes classification model 
from sklearn.naive_bayes import GaussianNB
#creating a model
gnb_model = GaussianNB()
#training the model
gnb_model.fit(predictor_variables,target_variable)
#test the model
prediction = gnb_model.predict([[0,2]])
print("\n")
print("Input given to the trained model is (0,2) i.e weather condition is overcast and tempreture is mild ")
print("Model Predicted:",prediction)