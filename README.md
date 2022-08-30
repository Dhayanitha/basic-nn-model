# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![WhatsApp Image 2022-08-29 at 8 35 46 AM](https://user-images.githubusercontent.com/75235032/187334902-48776795-1773-49d3-997e-abae99b637ae.jpeg)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model with hidden layer 1 - 8 neurons , hidden layer 2 - 10 neurons and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data=pd.read_csv("dataset1.csv")
data.head()
x=data[['Input']].values
x
y=data[['Output']].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
Scaler.fit(x_test)
x_train1=Scaler.transform(x_train)
x_train1
x_train
AI_BRAIN=Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])
AI_BRAIN.compile(optimizer='rmsprop', loss='mse')
AI_BRAIN.fit(x_train1,y_train,epochs=2000)
loss_df=pd.DataFrame(AI_BRAIN.history.history)
loss_df.plot()
x_test1=Scaler.transform(x_test)
x_test1
AI_BRAIN.evaluate(x_test1,y_test)
x_n1=[[25]]
x_n1_1=Scaler.transform(x_n1)
AI_BRAIN.predict(x_n1_1)
```
## Dataset Information

![D4](https://user-images.githubusercontent.com/75235032/187334728-17957598-3d5f-4d10-b543-706dab4122b0.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![D1](https://user-images.githubusercontent.com/75235032/187334588-5939c704-d6b9-4fb4-97e2-945d3b483007.png)

### Test Data Root Mean Squared Error

![D2](https://user-images.githubusercontent.com/75235032/187334606-f926eddd-c851-463c-a388-99e97ba9096b.png)

### New Sample Data Prediction

![D3](https://user-images.githubusercontent.com/75235032/187334623-c4a9c09f-8e6a-4101-8281-10f183ef6d3d.png)

## RESULT
Thus,the neural network regression model for the given dataset is developed.
