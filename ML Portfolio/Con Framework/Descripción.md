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
![image](https://user-images.githubusercontent.com/69470979/188771701-8dae8948-d233-4dff-a129-77955709e6a9.png)


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
![image](https://user-images.githubusercontent.com/69470979/188771922-7d34a6cf-3a27-4e02-8f97-bdbb26c6d91d.png)


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
  
# QUALITY OF THE PREDICTIONS:

En lo que respecta al MSE, esta métrica muestra el promedio de los cuadrados de los errores por lo que, aunque esta métrica permite medir el nivel de cambio lineal 
que ocurre en los datos, el hecho de elevar las diferencias al cuadrado tiende a magnificar/inflar los errores y por ende, datos muy alejados logran impactar 
mayormente al resultado general. Por tal motivo, se utiliza el RMSE el cuál quita este efecto de inflación al obtener la raíz cuadrada del MSE. Tal fue el caso del 
modelo de Orden 2 ya que si bien los MSE's de entrenamiento y validación estuvieron por debajo de 0.9, al obtener la métrica de los RMSE's estos valores fueron aún 
más bajos con una diferencia de 0.2 debido al efecto de la inflación del cuadrado de los errores, el cuál es un margen extremadamente bueno de error. Finalmente, se 
corroboraron estos valores con la métrica del MAE en la cuál las puntuaciones aumentan linealmente con el error promedio de los valores del error absoluto. Por ende, 
los valores del MAE coincidieron con los del RMSE lo que quiere decir que independientemente del signo de los errores, estos se encuentrán por debajo de 0.6, 
lo que indica un alto nivel de precisión en las predicciones. 

Dentro del contexto del problema, este error es relevante ya que lo que se está intentando predecir es el nivel de alcohol que tiene un vino dadas su nivel de 
acidez y su pH, y dado que el nivel de alcohol oscila entre un valor de 8 a 14.2 con una desviación estándar de 1.23, es conveniente que nuestro modelo tenga un 
error mucho menor al valor de desviación estándar para tener una precisión alta y que los valores de alcohol que se obtengan en futuras observaciones sean lo más 
cercanos a los reales.

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
