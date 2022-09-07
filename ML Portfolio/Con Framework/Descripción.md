This is my portfolio for the Machine Learning Module of my Advanced Artificial Inteligence for Data Science course!

métrica de desempeño (valor logrado sobre el subset de prueba), predicciones de prueba (entradas, valor esperado, valor obtenido

# TASK DESCRIPTION:

In this deliverable I implemented a Machine Learning (ML) classification Algorithm with the use of a Machine Learning and/or estadistical framework/library to determine if a patient may have or not a stroke. For the development of the testing part, I generated 2  models based on desicion tree classification, each one using a different performance metric. Tree 1 uses entropy as its main performance metric to weight out each leaf, while Tree 2 uses the gini coefficient to weight out each leaf. In the preparation part, I made sure to separate the dataset in two groups as training and testing in order to test the precision of the algorithm with real results. Then, I tested the implementation of the model with the validation portion of my dataset and printed out some predictions as a sample. 

# LIBRARY USED:
	SKlearn - Desicion Tree

# DATASET USED: 

  **Name:** Brain stroke prediction dataset 
  	full_filled_stroke_data.csv
  
  **Source:** [Dataset en Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
  **Original Source:** Data files © Original Authors
   
  **Length:** 4981

# VARIABLES:

  **Feature Variables (x):**
          * 1) gender: "Male", "Female" or "Other"
	  * 2) age: age of the patient
	  * 3) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
          * 4) heartdisease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease 
	  * 5) evermarried: "No" or "Yes"
	  * 6) worktype: "children", "Govtjov", "Neverworked", "Private" or "Self-employed" 7) Residencetype: "Rural" or "Urban"
          * 8) avgglucoselevel: average glucose level in blood
          * 9) bmi: body mass index
          * 10) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*

  **Predictor Variable (y):**
          * 1 if the patient had a stroke or 0 if not

# PREDICTIONS:

**Tree 1 with Entropy Metric:**

| index | gender | age | hypertension | heartdisease | evermarried | worktype | avgglucoselevel | bmi | smoking_status | real stroke | pred stroke | compare
| ------------- | ------------- | ------------- |------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
635|1|19|0|0|0|1|1|91.69|39.5|2|0|0|TRUE|
796|0|80|0|0|1|1|1|56.99|26.7|0|0|0|TRUE|
572|0|28|0|0|0|1|1|85.79|26.7|2|0|0|TRUE|
307|1|32|0|0|1|1|0|98.09|25.2|4|0|0|TRUE|
4396|0|62|0|0|0|0|1|75.07|30.5|0|0|0|TRUE|
3310|0|27|0|0|0|0|1|65.12|41.1|4|0|0|TRUE|
3574|1|45|0|0|1|1|1|106.95|33.4|2|0|0|TRUE|
2769|0|12|0|0|0|4|0|111.47|32.3|0|0|0|TRUE|
2446|1|34|0|0|1|1|1|94.44|34.2|2|0|0|TRUE|
4288|0|14|0|0|0|1|0|72.88|26.5|0|0|0|TRUE|
3639|1|2|0|0|0|4|1|70.25|17|2|0|0|TRUE|
3629|0|65|1|0|1|1|1|79.17|29.6|2|0|0|TRUE|
1707|1|47|0|0|1|1|1|75.43|36.4|4|0|0|TRUE|
3434|0|38|0|0|0|1|1|162.72|31.9|4|0|0|TRUE|
1127|0|8|0|0|0|4|1|115.54|28.5|2|0|0|TRUE|
4231|0|10|0|0|0|4|1|69.2|23.5|1|0|0|TRUE|
1126|0|10|0|0|0|4|1|93.11|14.6|2|0|0|TRUE|
1188|1|80|0|0|1|3|1|72.61|27.6|0|0|0|TRUE|
3000|1|54|0|0|1|1|1|109.27|43.8|1|0|0|TRUE|
3035|1|48|0|0|1|1|1|73.56|27.1|4|0|0|TRUE|
1362|1|39|0|0|0|0|1|87.33|34.3|0|0|0|TRUE|
4495|1|61|0|0|1|3|1|69.77|29.9|0|0|0|TRUE|
2374|1|35|0|0|0|1|0|71.59|40.3|0|0|0|TRUE|
4965|0|52|1|0|1|0|1|116.62|31.7|4|0|0|TRUE|
1453|1|77|0|0|1|1|1|93.48|25.2|1|0|0|TRUE|
2751|0|80|1|0|1|3|1|232.12|28.8|0|0|1|FALSE|
3357|0|17|0|0|0|1|1|83.26|32.9|0|0|0|TRUE|
3388|1|20|0|0|0|1|0|66.55|26.9|4|0|0|TRUE|
3172|1|47|0|0|1|1|0|77.91|30.3|1|0|0|TRUE|
32|0|58|0|0|0|1|1|92.62|32|2|1|0|FALSE|
2807|1|57|0|0|1|0|1|87.1|48.3|4|0|1|FALSE|
2608|0|20|0|0|0|1|1|88.47|28.1|4|0|0|TRUE|
4954|1|79|1|0|1|3|1|92.43|29.2|0|0|0|TRUE|
3560|1|40|0|0|1|1|0|72.12|38|0|0|0|TRUE|
3093|1|41|0|0|1|1|1|93.67|35.9|2|0|0|TRUE|
1340|1|2|0|0|0|4|1|65.96|19.7|2|0|0|TRUE|
4200|1|39|0|0|0|1|1|90.11|23.6|0|0|0|TRUE|
4840|0|13|0|0|0|4|0|71.73|22.6|2|0|0|TRUE|
3842|0|37|0|0|1|0|1|80.2|30.9|0|0|0|TRUE|
1484|1|81|0|1|1|1|0|84.93|31.8|2|0|0|TRUE|
3743|1|38|0|0|1|1|0|103.58|30.8|1|0|1|FALSE|
1884|0|8|0|0|0|4|0|104.51|20.6|2|0|0|TRUE|
1937|1|43|0|0|1|3|0|75.77|20.4|1|0|0|TRUE|
3019|1|71|0|0|1|1|1|91.85|27.6|1|0|0|TRUE|
4369|1|66|0|0|1|3|0|102.73|35|1|0|0|TRUE|
3318|0|38|0|0|1|1|0|86.93|31.1|0|0|0|TRUE|
3495|1|62|0|0|1|1|0|101.19|23.4|0|0|0|TRUE|
3253|0|14|0|0|0|1|1|164.7|26.3|2|0|0|TRUE|
1932|1|53|0|0|1|3|1|72.49|38.5|0|0|0|TRUE|
128|1|55|0|0|1|3|1|92.98|25.6|0|1|0|FALSE|
2933|1|75|0|0|1|1|1|108.72|29.2|1|0|0|TRUE|
491|1|76|0|0|1|1|0|183.34|39.5|1|0|0|TRUE|
3897|1|78|0|0|1|1|0|119.13|25|0|0|0|TRUE|
200|0|66|0|0|1|1|1|76.46|21.2|1|1|0|FALSE|
1855|0|59|0|0|1|0|1|96.25|23.3|1|0|0|TRUE|
1108|0|21|0|0|0|1|0|82.71|20.1|1|0|0|TRUE|
938|1|55|0|0|1|1|1|109.59|26.2|1|0|0|TRUE|
4506|1|63|1|0|1|1|1|57.15|38.8|0|0|1|FALSE|
4097|1|79|0|0|1|3|1|65.58|26.1|2|0|0|TRUE|
4225|0|82|0|1|1|1|0|57.56|27.5|0|0|0|TRUE|
2556|0|3|0|0|0|4|0|88.43|17.7|2|0|0|TRUE|
563|1|40|0|0|1|1|0|71.2|27.1|0|0|0|TRUE|
4406|1|19|0|0|0|1|1|66.7|24.7|0|0|0|TRUE|
4548|0|5|0|0|0|4|1|101.31|20|2|0|0|TRUE|
425|1|53|0|0|1|3|1|96.88|31.4|2|0|0|TRUE|
650|1|81|0|0|1|3|0|90.9|31.2|1|0|0|TRUE|
2688|1|56|0|0|1|1|0|94.19|25.7|0|0|0|TRUE|
724|1|23|0|0|0|1|0|79.39|27.6|0|0|0|TRUE|
625|1|45|0|0|1|1|1|89.21|21.6|1|0|1|FALSE|
1308|1|41|0|0|1|1|0|80.77|21.1|0|0|0|TRUE|
3298|1|80|1|0|1|1|0|125.89|28.9|4|0|0|TRUE|
3064|0|60|0|0|1|1|0|74.08|35.9|2|0|1|FALSE|
3578|1|42|0|0|1|0|0|99.94|33.4|0|0|0|TRUE|
4638|1|2|0|0|0|4|1|80.3|21.2|2|0|0|TRUE|
1074|1|78|0|0|0|1|0|67.96|26.8|2|0|1|FALSE|
3770|0|55|0|0|1|0|0|231.15|22.3|0|0|0|TRUE|
1971|0|24|0|0|0|1|0|72.29|22.2|2|0|0|TRUE|
1721|1|53|0|0|1|3|0|113.74|31.6|4|0|0|TRUE|
691|1|31|0|0|1|1|0|106.18|27|4|0|0|TRUE|
1859|0|11|0|0|0|4|0|99.79|20.2|2|0|0|TRUE|
2714|0|1.64|0|0|0|4|0|170.88|20.8|2|0|0|TRUE|
1590|1|69|0|0|1|3|1|63.19|32.2|0|0|0|TRUE|
1891|1|50|0|0|1|0|0|82.37|30.7|0|0|0|TRUE|
3457|1|16|0|0|0|4|0|64.51|21.2|2|0|0|TRUE|
874|0|24|0|0|0|1|0|59.28|43.2|0|0|0|TRUE|
2588|1|39|0|0|1|0|1|69.38|22.1|2|0|0|TRUE|
281|1|57|1|0|1|1|1|235.85|40.1|0|0|0|TRUE|
1134|1|56|0|0|1|3|0|124.16|23|0|0|0|TRUE|
1904|1|15|0|0|0|1|0|121.6|22.8|0|0|0|TRUE|
4236|0|4|0|0|0|4|0|87|19|2|0|0|TRUE|
4026|0|15|0|0|0|4|1|78.9|23|2|0|0|TRUE|
847|0|80|0|0|1|3|1|236.84|26.8|0|0|0|TRUE|
492|0|52|0|0|1|1|0|247.69|35.1|2|0|0|TRUE|
880|0|21|0|0|0|1|1|102.05|29.9|0|0|0|TRUE|
3820|1|62|0|0|1|1|0|212.62|35.8|0|0|0|TRUE|
3281|1|71|0|0|1|1|0|80.34|29.2|0|0|0|TRUE|
1562|1|44|0|0|0|0|0|215.9|41.8|4|0|0|TRUE|
3053|1|67|0|0|1|3|0|110.41|28.7|0|0|0|TRUE|
4706|1|20|0|0|0|1|1|117.59|17.1|0|0|0|TRUE|
1118|1|78|1|0|1|3|1|59.2|29.1|2|0|0|TRUE|


**Tree 2 with Gini Metric:**
| index | gender | age | hypertension | heartdisease | evermarried | worktype | avgglucoselevel | bmi | smoking_status | real stroke | pred stroke | compare
| ------------- | ------------- | ------------- |------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
635|1|19|0|0|0|1|1|91.69|39.5|2|0|0|TRUE|
796|0|80|0|0|1|1|1|56.99|26.7|0|0|0|TRUE|
572|0|28|0|0|0|1|1|85.79|26.7|2|0|0|TRUE|
307|1|32|0|0|1|1|0|98.09|25.2|4|0|0|TRUE|
4396|0|62|0|0|0|0|1|75.07|30.5|0|0|0|TRUE|
3310|0|27|0|0|0|0|1|65.12|41.1|4|0|0|TRUE|
3574|1|45|0|0|1|1|1|106.95|33.4|2|0|0|TRUE|
2769|0|12|0|0|0|4|0|111.47|32.3|0|0|0|TRUE|
2446|1|34|0|0|1|1|1|94.44|34.2|2|0|0|TRUE|
4288|0|14|0|0|0|1|0|72.88|26.5|0|0|0|TRUE|
3639|1|2|0|0|0|4|1|70.25|17|2|0|0|TRUE|
3629|0|65|1|0|1|1|1|79.17|29.6|2|0|0|TRUE|
1707|1|47|0|0|1|1|1|75.43|36.4|4|0|0|TRUE|
3434|0|38|0|0|0|1|1|162.72|31.9|4|0|0|TRUE|
1127|0|8|0|0|0|4|1|115.54|28.5|2|0|0|TRUE|
4231|0|10|0|0|0|4|1|69.2|23.5|1|0|0|TRUE|
1126|0|10|0|0|0|4|1|93.11|14.6|2|0|0|TRUE|
1188|1|80|0|0|1|3|1|72.61|27.6|0|0|0|TRUE|
3000|1|54|0|0|1|1|1|109.27|43.8|1|0|0|TRUE|
3035|1|48|0|0|1|1|1|73.56|27.1|4|0|0|TRUE|
1362|1|39|0|0|0|0|1|87.33|34.3|0|0|0|TRUE|
4495|1|61|0|0|1|3|1|69.77|29.9|0|0|0|TRUE|
2374|1|35|0|0|0|1|0|71.59|40.3|0|0|0|TRUE|
4965|0|52|1|0|1|0|1|116.62|31.7|4|0|0|TRUE|
1453|1|77|0|0|1|1|1|93.48|25.2|1|0|0|TRUE|
2751|0|80|1|0|1|3|1|232.12|28.8|0|0|0|TRUE|
3357|0|17|0|0|0|1|1|83.26|32.9|0|0|0|TRUE|
3388|1|20|0|0|0|1|0|66.55|26.9|4|0|0|TRUE|
3172|1|47|0|0|1|1|0|77.91|30.3|1|0|0|TRUE|
32|0|58|0|0|0|1|1|92.62|32|2|1|0|FALSE|
2807|1|57|0|0|1|0|1|87.1|48.3|4|0|1|FALSE|
2608|0|20|0|0|0|1|1|88.47|28.1|4|0|0|TRUE|
4954|1|79|1|0|1|3|1|92.43|29.2|0|0|0|TRUE|
3560|1|40|0|0|1|1|0|72.12|38|0|0|0|TRUE|
3093|1|41|0|0|1|1|1|93.67|35.9|2|0|0|TRUE|
1340|1|2|0|0|0|4|1|65.96|19.7|2|0|0|TRUE|
4200|1|39|0|0|0|1|1|90.11|23.6|0|0|0|TRUE|
4840|0|13|0|0|0|4|0|71.73|22.6|2|0|0|TRUE|
3842|0|37|0|0|1|0|1|80.2|30.9|0|0|0|TRUE|
1484|1|81|0|1|1|1|0|84.93|31.8|2|0|0|TRUE|
3743|1|38|0|0|1|1|0|103.58|30.8|1|0|0|TRUE|
1884|0|8|0|0|0|4|0|104.51|20.6|2|0|0|TRUE|
1937|1|43|0|0|1|3|0|75.77|20.4|1|0|0|TRUE|
3019|1|71|0|0|1|1|1|91.85|27.6|1|0|0|TRUE|
4369|1|66|0|0|1|3|0|102.73|35|1|0|0|TRUE|
3318|0|38|0|0|1|1|0|86.93|31.1|0|0|0|TRUE|
3495|1|62|0|0|1|1|0|101.19|23.4|0|0|0|TRUE|
3253|0|14|0|0|0|1|1|164.7|26.3|2|0|0|TRUE|
1932|1|53|0|0|1|3|1|72.49|38.5|0|0|0|TRUE|
128|1|55|0|0|1|3|1|92.98|25.6|0|1|0|FALSE|
2933|1|75|0|0|1|1|1|108.72|29.2|1|0|0|TRUE|
491|1|76|0|0|1|1|0|183.34|39.5|1|0|1|FALSE|
3897|1|78|0|0|1|1|0|119.13|25|0|0|1|FALSE|
200|0|66|0|0|1|1|1|76.46|21.2|1|1|0|FALSE|
1855|0|59|0|0|1|0|1|96.25|23.3|1|0|0|TRUE|
1108|0|21|0|0|0|1|0|82.71|20.1|1|0|0|TRUE|
938|1|55|0|0|1|1|1|109.59|26.2|1|0|0|TRUE|
4506|1|63|1|0|1|1|1|57.15|38.8|0|0|0|TRUE|
4097|1|79|0|0|1|3|1|65.58|26.1|2|0|0|TRUE|
4225|0|82|0|1|1|1|0|57.56|27.5|0|0|0|TRUE|
2556|0|3|0|0|0|4|0|88.43|17.7|2|0|0|TRUE|
563|1|40|0|0|1|1|0|71.2|27.1|0|0|0|TRUE|
4406|1|19|0|0|0|1|1|66.7|24.7|0|0|0|TRUE|
4548|0|5|0|0|0|4|1|101.31|20|2|0|0|TRUE|
425|1|53|0|0|1|3|1|96.88|31.4|2|0|0|TRUE|
650|1|81|0|0|1|3|0|90.9|31.2|1|0|0|TRUE|
2688|1|56|0|0|1|1|0|94.19|25.7|0|0|0|TRUE|
724|1|23|0|0|0|1|0|79.39|27.6|0|0|0|TRUE|
625|1|45|0|0|1|1|1|89.21|21.6|1|0|0|TRUE|
1308|1|41|0|0|1|1|0|80.77|21.1|0|0|0|TRUE|
3298|1|80|1|0|1|1|0|125.89|28.9|4|0|0|TRUE|
3064|0|60|0|0|1|1|0|74.08|35.9|2|0|0|TRUE|
3578|1|42|0|0|1|0|0|99.94|33.4|0|0|0|TRUE|
4638|1|2|0|0|0|4|1|80.3|21.2|2|0|0|TRUE|
1074|1|78|0|0|0|1|0|67.96|26.8|2|0|0|TRUE|
3770|0|55|0|0|1|0|0|231.15|22.3|0|0|0|TRUE|
1971|0|24|0|0|0|1|0|72.29|22.2|2|0|0|TRUE|
1721|1|53|0|0|1|3|0|113.74|31.6|4|0|0|TRUE|
691|1|31|0|0|1|1|0|106.18|27|4|0|0|TRUE|
1859|0|11|0|0|0|4|0|99.79|20.2|2|0|0|TRUE|
2714|0|1.64|0|0|0|4|0|170.88|20.8|2|0|0|TRUE|
1590|1|69|0|0|1|3|1|63.19|32.2|0|0|0|TRUE|
1891|1|50|0|0|1|0|0|82.37|30.7|0|0|0|TRUE|
3457|1|16|0|0|0|4|0|64.51|21.2|2|0|0|TRUE|
874|0|24|0|0|0|1|0|59.28|43.2|0|0|0|TRUE|
2588|1|39|0|0|1|0|1|69.38|22.1|2|0|0|TRUE|
281|1|57|1|0|1|1|1|235.85|40.1|0|0|0|TRUE|
1134|1|56|0|0|1|3|0|124.16|23|0|0|0|TRUE|
1904|1|15|0|0|0|1|0|121.6|22.8|0|0|0|TRUE|
4236|0|4|0|0|0|4|0|87|19|2|0|0|TRUE|
4026|0|15|0|0|0|4|1|78.9|23|2|0|0|TRUE|
847|0|80|0|0|1|3|1|236.84|26.8|0|0|1|FALSE|
492|0|52|0|0|1|1|0|247.69|35.1|2|0|0|TRUE|
880|0|21|0|0|0|1|1|102.05|29.9|0|0|0|TRUE|
3820|1|62|0|0|1|1|0|212.62|35.8|0|0|1|FALSE|
3281|1|71|0|0|1|1|0|80.34|29.2|0|0|0|TRUE|
1562|1|44|0|0|0|0|0|215.9|41.8|4|0|0|TRUE|
3053|1|67|0|0|1|3|0|110.41|28.7|0|0|0|TRUE|
4706|1|20|0|0|0|1|1|117.59|17.1|0|0|0|TRUE|
1118|1|78|1|0|1|3|1|59.2|29.1|2|0|0|TRUE|


# TRAINING-VALIDATE DATA SEPARATION:

  For the model implementation, it was necessary to separate the dataset into two subsets:

  * Training set
  * Test set
      
   For that reason a ration of 60:40 was considered to divide the dataset using the train_test_split module from the sklearn.model_selection library. Meaning that 2988 data were asigned to the training set and 1992 data were assigned to the testing set.

# TRAINING QUALITY:

 
  
# QUALITY OF THE PREDICTIONS:

  * ***Confusion Matrix:***

# PERFORMANCE METRICS:
  * ***Accuracy:***
Is one of the most common performance metrics to evaluate a classification model. In simple terms, it represents the fraction of correct predictions that the model got right as a percentage of the total predictions that are made. It is calculated as the ration between the number of correct predictions over the total number of predictions. *Within everything that has been predicted as a positive, precision counts the percentage that is correct* In case of a binary classification, accuracy can  be calculated in terms of positives and negatives as follows: 

![image](https://user-images.githubusercontent.com/69470979/188773515-17e431ab-f949-4d26-b0af-f54c2ed73108.png)

Where TP = True Positives, TN = True Negatives, FP = False Positives, and FN = False Negatives. 
Accuracy may be one of the main evaluation metrics, but it doesn't always tell us the whole truth, specially when  you're working with a class-imbalanced data set where there is a significant disparity between the number of positive and negative labels.

  * ***Recall:***
Recall is the second metric one should look out for after measuring accuracy, since it gives us insight about the model's performance when it comes to the proportion of actual positives that were correctly identified. *Within everything that actually is positive, how many did the model succeed to find.* It is calculated as follows:

![image](https://user-images.githubusercontent.com/69470979/188773942-663ba216-a1f7-48cf-846f-5ad480e9f4d1.png)

Where TP = True Positives and FN = False Negatives. 
    
  * ***F1 Score:***
The F1 score represents an improvement of two simpler performance metrics: precision and recall. In the majority of cases, it is possible to modify a model to increase precision at a cost of a lower recall, or on the other hand increase recall at the cost of lower precision. Therefore, the goal of the F1 Score is to combine both of this metrics into a single one, in order to correctly analize unbalanced data. Hence, the F1 score is defined as the harmonic mean of precision and recall. 


# GOOGLE COLAB URL:
https://colab.research.google.com/drive/1ucEgN-YAAssPGy2eJw1gXxcQlkAERKiZ?usp=sharing
