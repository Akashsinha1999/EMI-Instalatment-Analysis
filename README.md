
# EMI Installment Analysis using Deep Learning

This project focused on predicting whether a person will pay his/her EMI (Equated Monthly Installment) in the next month using deep learning techniques. The project utilizes a Sequential Model to analyze the data and make predictions.

## Project Overview
In today's financial landscape, assessing the likelihood of EMI repayments is crucial for lending institutions. This project aims to create a predictive model that can assist in making informed decisions regarding EMI payments.
## Data
The project utilizes a dataset containing historical information about borrowers, including demographic details such as age , gender , education ,sex , marriage, financial history like account balance , card limit, and previous EMI repayment records.

Variables in Data are -
ID,	LIMIT_BAL,	SEX,	EDUCATION,	MARRIAGE,  AGE,	PAY_0 ,PAY_1 ,PAY_2	,PAY_3,	PAY_4,	PAY_5,	PAY_6,	BILL_AMT1,	BILL_AMT2,	BILL_AMT3,	BILL_AMT4,	BILL_AMT5,	BILL_AMT6,	PAY_AMT1,	PAY_AMT2,	PAY_AMT3,	PAY_AMT4,	PAY_AMT5,	PAY_AMT6,	default.payment.next.month

ID = 1,2,3....

LIMIT_BAL mean how much a person can spent 


SEX - Male , Female there on data it is represented as 1,2

Marriage - whether a person is married or not 

Age - Borrower age 

PAY_0,PAY_2	,PAY_3,	PAY_4,	PAY_5,	PAY_6 - Last six month pay time whether a person paid emi on time or not ,if yes then it is denoted as 0 ,if person paid 2 days before the emi submission date then it is denoted as -2 and if a persond paid after 1 day of the submission date then it is denoted as 1.


BILL_AMT1,	BILL_AMT2,	BILL_AMT3,	BILL_AMT4,	BILL_AMT5,	BILL_AMT6, -   how much amount borrowers had to pay in last 6 month emi.

PAY_AMT1,	PAY_AMT2,	PAY_AMT3,	PAY_AMT4,	PAY_AMT5,	PAY_AMT6 - borrowers has paid how much amount as emi in previous 6 month.







## POC / Technique 
I have imported Pandas , NumpY for data wrangling such as cleaning , droping nan values.imported matplotlib , seaborn for data visualization and to know correlation between the variables.

Deep learning approach is slightly different then a machine learning approach , 
ML approach 
ML Approach
1. Split x, y 
2. Get train data and test data ( x_train, x_test, y_train, y_test) 
3. get model and get its object and set the hyperparameters
             from sklearn.ensembple import RandomForestClassifier
             rf=RandomForestClassifier(max_depth=12, n_estimator=100, min_sample_split=30)
4. fit the model 
             rf.fit(x_train, y_train)
5. Evaluate the model
              rf.score(x_train, y_train)
              confusion matrix, accuracy recall , etc 


DL approach   

1. Split x, y 
2. Get train data and test data ( x_train, x_test, y_train, y_test,x_val, y_val) 
3. get the model and make its object 



4 from keras.model import Sequential                                                  - 

importing model from keras package

 model=Sequential()                                                                       - creating object for the model

         model.add(Dense(units=8,activation="linear",kernel_initializer="uniform",input_dim=11 )) - input Layer ( input_dim= No of x variables; units- how many neurons )
         model.add(Dense(units=16,activation="linear",kernel_initializer="uniform"))              - hidden layer
         model.add(Dense(units=32,activation="linear",kernel_initializer="uniform"))              - hidden layer
         model.add(Dense(units=16,activation="linear",kernel_initializer="uniform"))              - hidden layer
         model.add(Dense(units=1, activation="sigmoid",kernel_initializer="uniform" ))             - Output Layer                                                          
         
 Output Layer  ( units/neurons in this layer depends on the class of the target variable- if my target )
                                                          
 compilation step  
          
          model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=["accuracy"])           - Compilation stage

5. train the model/ fit the model
          
             model.fit(x_train, y_train , epoch=10, batch_size=32, validation =(x_val, y_val))          - Fitting the model 
6. Evaluate the model 
      
      Evaluation of the model is same as ML models   


A Sequential Deep Learning model is implemented in this project to learn from the data and make predictions. The model is trained on the historical dataset to classify whether an individual is likely to pay their EMI in the upcoming month.




## Result
The project provides insights into the predictive capabilities of the deep learning model, which can be invaluable for risk assessment in the lending industry.
## Usage
You can use this repository to explore the code, data, and model implementation. Follow the instructions in the documentation to replicate the analysis and make your predictions based on your own data with 77% od accuracy.


## Contributor

AKASH KUMAR SINHA 

Feel free to contribute to this project by opening issues, suggesting improvements, or submitting pull requests.
