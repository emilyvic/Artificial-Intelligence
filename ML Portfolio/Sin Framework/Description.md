This is my portfolio for the Machine Learning Module of my Advanced Artificial Inteligence for Data Science course!

# TASK DESCRIPTION:

In this deliverable I implemented a Machine Learning (ML) Algorithm without the use of a Machine Learning and/or estadistical framework/library to determine the alcohol quantity in a wine sample. For the development of the testing part, I generated 3 models based on linear regression, each one with a different degree equation. Being 1-degree, 2-dregree and 4-degree algorithms respectively. In the preparation part, I made sure to separate the dataset in two groups as training and validation in order to test the precision of the  algorithm with real results. Then, I tested the implementation of the model with the validation portion of my dataset and printed out some predictions as a sample. 

# DATASET USED: 

  **Name:** Winequality.csv
  
  **Source:** https://archive.ics.uci.edu/ml/datasets/wine+quality
  
  **Original Source:** Paulo Cortez, University of Minho, Guimarães, Portugal, http://www3.dsi.uminho.pt/pcortez
    A. Cerdeira, F. Almeida, T. Matos and J. Reis, Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal
    @2009
   
  **Length:** 4898

# VARIABLES:

  **1-degree Model:** 
  
  Feature Variable (x):
          * x1 = Acidity level

   Predictor Variable (y):
          * Alcohol level

  **2-degree Model:**
  
   Feature Variable (x):
          * x1 = Acidity level
          * x2 = pH

   Predictor Variable (y):
          * Alcohol level
	  
  **4-degree Model:**
        Feature Variable (x):
          * x1 = Acidity level
          * x2 = pH
          * x3 = sulphates
          * x4 = chlorides

  Predictor Variable (y):
          * Alcohol level

# PREDICTIONS:

**Model 1:**

| volatile_acidity | pred_alcohol | real_alcohol | dif |
| ------------- | ------------- | ------------- |------------- |
0.35|12.682355|14.2|1.517645|
0.31|12.244429|13.2|0.955571|
0.25|11.587539|11.2|0.387539|
0.36|12.791837|10.2|2.591837|
0.31|12.244429|12.8|0.555571|
0.31|12.244429|10.1|2.144429|
0.31|12.244429|10.1|2.144429|
0.22|11.259094|10.3|0.959094|
0.14|10.383241|9.9|0.483241|
0.22|11.259094|11.5|0.240906|
0.32|12.35391|9|3.35391|
0.32|12.35391|9|3.35391|
0.28|11.915984|12.1|0.184016|
0.3|12.134947|13.3|1.165053|

**Model 2:**
| volatile_acidity | pH | pred_alcohol | real_alcohol | dif |
| ------------- | ------------- | ------------- |------------- | ------------- |
0.35|3.12|10.358484|14.2|3.841516|
0.31|3|9.998539|13.2|3.201461|
0.25|3.04|10.118522|11.2|1.081478|
0.36|3.14|10.418475|10.2|0.218475|
0.31|3.09|10.268499|12.8|2.531501|
0.31|2.95|9.848561|10.1|0.251439|
0.31|2.95|9.848561|10.1|0.251439|
0.22|2.83|9.488616|10.3|0.811384|
0.14|3.19|10.568459|9.9|0.668459|
0.22|3.15|10.448474|11.5|1.051526|
0.32|3.18|10.538458|9|1.538458|
0.32|3.18|10.538458|9|1.538458|
0.28|3.03|10.088526|12.1|2.011474|
0.3|3.14|10.418477|13.3|2.881523|

**Model 4:**
| volatile_acidity | pH | sulphates | chlorides | pred_alcohol | real_alcohol | dif |
| ------------- | ------------- | ------------- |------------- | ------------- | ------------- | ------------- |
0.35|3.12|0.4|0.037|8.588462|14.2|5.611538|
0.31|3|0.4|0.035|8.294482|13.2|4.905518|
0.25|3.04|0.53|0.05|9.771173|11.2|1.428827|
0.36|3.14|0.57|0.057|10.599866|10.2|0.399866|
0.31|3.09|0.36|0.038|8.422215|12.8|4.377785|
0.31|2.95|0.39|0.045|8.852826|10.1|1.247174|
0.31|2.95|0.39|0.045|8.852826|10.1|1.247174|
0.22|2.83|0.31|0.031|7.408369|10.3|2.891631|
0.14|3.19|0.33|0.056|9.404861|9.9|0.495139|
0.22|3.15|0.31|0.046|8.713697|11.5|2.786303|
0.32|3.18|0.56|0.063|10.951008|9|1.951008|
0.32|3.18|0.56|0.063|10.951008|9|1.951008|
0.28|3.03|0.41|0.03|8.010413|12.1|4.089587|
0.3|3.14|0.41|0.036|8.534017|13.3|4.765983|

			
# TRAINING-VALIDATE DATA SEPARATION:

  For the model implementation, it was necessary to separate the dataset into two subsets:

  * Training set
  * Validation set
      
   For that reason a ration of 80:20 was considered to divide the dataset. Meaning that 3918 data were asigned to the training set 
   and 980 data were assigned to the validation set.

# TRAINING QUALITY:

The best model implemented to predict the alcohol level in the wine sample was the 2-degree linear regression model since it had the best level of accuracy
from the other models implemented. Meaning that the predictions made by this model in particular were closer to the real data in comparison to the one´s made
with other models. The performance metric used to measure this accuracy level was the Mean Square Error (MSE). The feature variables used in this model in particular 
were:

* Acidity level
* pH level

  *Loss Function:*
  A loss function in Machine Learning is a measure of how accurately the  ML model is able to predict the expected outcome (the ground truth). 
  The loss function will take two items as input: 
  * the output value of our model 
  * the ground truth expected value. 
  * The output of the loss function is called the loss which is a measure of how well our model did at predicting the outcome. A high value for the loss means the model performed very poorly. A low value for the loss means our model performed very well.
  *  For the three models implemented, the loss function output was minimun as the initial values of theta were changed in order to minimize the error. 

## Training Metrics Table:

* **Training Metrics**

| Metric | Model 1 | Model 2 | Model 4 | 
| ------------- | ------------- | ------------- | ------------- |
| MSE | 2.29 | 0.80 | 2.97 | 
| RMSE | 0.84 | 0.54 | 0.89 | 
| MAE | 0.84 | 0.54 | 0.89 | 
  
  * **Validation Metrics**

| Metric | Model 1 | Model 2 | Model 4 | 
| ------------- | ------------- | ------------- | ------------- |
| MSE | 1.99 | 0.76 | 3.11 | 
| RMSE | 0.78 | 0.51 | 0.94 | 
| MAE | 0.78 | 0.51 | 0.94 | 

# QUALITY OF THE PREDICTIONS:

Concerning the MSE, this performance metric shows the square mean of the current errors. While this metric allows to measure the lineal variations on the data, the effect of the square magnifies the errors. Therefore, the furthest data have a heavier effect in the overall assesment result upon the model quality. Hence, the use of the RMSE which decreases the inflation effect by rooting the MSE. This was the case of the 2nd-grade model. Even that both training and validation MSE´s were below 0.9, by square-rooting these values were even lower with a 0.2 difference from the original MSE's, showing a final error (without inflation effect) of 0.51 and 0.54 respectively. These final results, were in fact, a great minimal error value which indicate a high accuracy in the regression model. 
Finally, this ciphers were compared to the MAE metric, in which scoring tend to increase in a lineal behaviour with the mean error. Subsecuently, the result of this comparison was that both RMSE and MAE were equivalent, which means that regardless of the sign of the errors, the error is below 0.6, indicating a high level of accuracy in the predictions made.

In the context of the problem, this error is relevant in the way that is 0.6 trustfull in predicting the alchohol level of a particular wine given it´s acidity and pH. Knowing that the alcohol level oscilates between 8-14.2 with a standard deviation of 1.23. The confidence level in the model can be justified upon the fact that the MSE value is less than the standard deviation. Thus, accuracy level can be considered as HIGH and that may guarantee that future observations and testing may be as closer to the actual values we are trying to predict.  

# PERFORMANCE METRICS:
  * ***MSE (Mean Square Error):***
  The MSE is an estimator that measures the average square error between the estimator and the prediction.
  Measures the difference between the prediction and the actual value of the distribution and is an accuracy measurement to
  determine how accurate the predictions were made based on how distant they were from the actual value. It takes into account the
  variance as well as the standard deviation of the dataset. It was for great value to make sure the dataset didn´t have many outlier predictions with
  a disproportional error, since the MSE puts on a great amount of weight into those errors.
  
  * ***RMSE (Root Mean Square Error):***
    Heuristically that RMSE it represents a normalized distance between the vector of predicted values and the vector of observed values that is being rescaled
    according to the size of observations. Since the MSE can sometimes increase the effect of the biggest errors, the RMSE is al alternative error to ignore the
    inflation effect from elevating the error to the square. Plus it helps to consider the standard deviation σ of a typical observed value from our model’s prediction, assuming that our observed data can be decomposed as:
    observed value = predicted value + predictably distributed random noise with mean zero.
    
  * ***MAE (Mean Absolute Error):***
    Absolute Error is the amount of error in your measurements. Is the mean of the absolute values of the individual prediction errors on over all instances in the test set. Each prediction error is the difference between the true value and the predicted value for the instance.

# GOOGLE COLAB URL:
https://colab.research.google.com/drive/1mR6XGK0iJwqSNgVdJQ8ejnUG2dT3PMeP?usp=sharing
