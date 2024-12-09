import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
data = pd.read_csv("Data_Set\\Housing.csv")
print(data.info()) 

x = data[["area","bedrooms","bathrooms"]] #Features for the training the models or independent varialble
y = data["price"] # Dependent variable

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42) #Spliting the data for training(80%) and testing(20%)

my_model = DecisionTreeRegressor() #Creating an instance for the Decision Tree Regessor

my_model.fit(x_train,y_train) #Training the model 

prediction = my_model.predict(x_test) #Predicting the data which were splited into train and test data

print(prediction)

# Predicting price for manual input data depending on the features "area", "bedrooms", "bathrooms"

new_data = pd.DataFrame({
    "area":[1234],
    "bedrooms":[2],
    "bathrooms":[2]
}) #Price will be predicted for these specification area=1234, bedrooms=2, bathrooms=2 // you can change the values to see how price will vary for differnt specification

new_pridiction = my_model.predict(new_data)
print(f"\n\n User input/speification for the house:\n{new_data}\n")
print(f"\nPrice for the user input/speification for the house:: {new_pridiction}\n")

#Evaluation of the model
mse = mean_squared_error(y_test,prediction)
r2 = r2_score(y_test,prediction)

print("\nMean_Squared_Error: ",mse)
print("\nR^2 score: ",r2)


# To find the accuracy for the model we can also use accuracy matrics


# Accuracy = metrics.accuracy_score(actual_value, predicted_value) #change the variables with their respected: (actual_value, predicted_value) 