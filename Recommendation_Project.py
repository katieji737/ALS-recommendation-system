
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


#importing the findspark and entering the path where the spark 2.3.0 is installed
import findspark
findspark.init()


# In[ ]:


#importing the pyspark 
import pyspark


# In[ ]:


#creating a spark session from the pyspark.sql
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# In[ ]:


#Using the jupyter it automatically creates the SparkContext(sc) and now we need to create the SQLContext with in the spark context
from pyspark.sql import SQLContext


# In[ ]:


#initializing the spark session and spark context
sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)


# In[ ]:


# load products info
products = sqlContext.read.load('products.csv',format='com.databricks.spark.csv',header='true',inferSchema='true')


# In[ ]:


#products.show(5)


# In[ ]:


#load order and product info
order_product1= sqlContext.read.load('order_products_5000.csv',format='com.databricks.spark.csv',header='true',inferSchema='true')


# In[ ]:


#order_product1.show(5)


# In[ ]:


# load department info
departments= sqlContext.read.load('departments.csv',format='com.databricks.spark.csv',header='true',inferSchema='true')


# In[ ]:


#departments.show(5)


# In[ ]:


#combine 3 tables

order_p=order_product1.join(products,["product_id"])

data=order_p.join(departments,['department_id'])


# In[ ]:


data.show(5)


# In[ ]:


data.registerTempTable("data")


# In[ ]:


#Rename the column order_id
data2 = sqlContext.sql("SELECT order_id AS user_id, product_id, product_name, rating, department_id,department,aisle_id from data")
data2.show()


# In[ ]:


data=data2
data.registerTempTable("data")


# ## Basic Recommendations

# In[ ]:


#when a new customer comes in and have no info about him/her, recommend the top5 popular products.
#for depart in departments: where department=depart

model_default=sqlContext.sql("select product_id, product_name, count('product_id') as cnt from data group by product_id, product_name order by cnt desc""")
model_default.take(5)


# # SVD&NMF&KNN

# In[ ]:


ratings = pd.read_csv('/Users/katie/Desktop/order_products_5000.csv')
ratings = ratings.drop(['Unnamed: 0','reordered'],axis=1)
ratings.head(5)


# In[ ]:


#surprise package
from surprise import Reader, Dataset


# In[ ]:


# to load dataset from pandas df, we need `load_fromm_df` method in surprise lib
ratings_dict = {'product_id': list(ratings.product_id),
                'user_id': list(ratings.order_id),
                'rating': list(ratings.rating)}
df = pd.DataFrame(ratings_dict)


# In[ ]:


# A reader is still needed but only the rating_scale param is required.
# The Reader class is used to parse a file containing ratings.
reader = Reader(rating_scale=(1.0, 10.0))


# In[ ]:


# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)


# In[ ]:


# Split data into 5 folds
data.split(n_folds=5)


# In[ ]:


from surprise import SVD, evaluate
from surprise import NMF
from surprise import KNNBasic


# In[ ]:


# svd
algo = SVD()
evaluate(algo, data, measures=['RMSE'])


# In[ ]:


# nmf
algo = NMF()
evaluate(algo, data, measures=['RMSE'])


# In[ ]:


# knn
algo = KNNBasic()
evaluate(algo, data, measures=['RMSE'])


# ## Collaborative Filtering

# In[ ]:


# Smaller dataset so we will use 0.7 / 0.1 / 0.2
seed=12345
(training,valid,test) = data.randomSplit([0.7,0.1,0.2],seed=seed)


# In[ ]:


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


# In[ ]:


# Let's initialize our ALS learner
als = ALS(nonnegative=True)


# In[ ]:


# Now set the parameters for the method
als.setMaxIter(5)   .setSeed(seed)   .setItemCol("product_id")   .setRatingCol("rating")   .setUserCol("user_id")


# In[ ]:


# Now let's compute an evaluation metric for our test dataset
# We Create an RMSE evaluator using the label and predicted columns
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="rmse")


# In[ ]:


tolerance = 0.03
ranks = [4]
regParams = [1]
errors = [[0]*len(ranks)]*len(regParams)
models = [[0]*len(ranks)]*len(regParams)
err = 0
min_error = float('inf')
best_rank = -1
i = 0


# In[ ]:


from pyspark.sql import functions as F


# ## Tune Parameters

# In[ ]:


for regParam in regParams:
    j = 0
    for rank in ranks:
    # Set the rank here:
        als.setParams(rank = rank, regParam = regParam)
    # Create the model with these parameters.
        model = als.fit(training)
    # Run the model to create a prediction. Predict against the validation_df.
        predict_df = model.transform(valid)

    # Remove NaN values from prediction (due to SPARK-14489)
        predicted_counts_df = predict_df.filter(predict_df.prediction != float('nan'))
        #predicted_counts_df = predicted_counts_df.withColumn("prediction", F.abs(F.round(predicted_counts_df["prediction"],0)))
    # Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
        error = reg_eval.evaluate(predicted_counts_df)
        errors[i][j] = error
        models[i][j] = model
        print ('For rank %s, regularization parameter %s the RMSE is %s' % (rank, regParam, error))
        if error < min_error:
            min_error = error
            best_params = [i,j]
        j += 1
    i += 1


# In[ ]:


als.setRegParam(regParams[best_params[0]])
als.setRank(ranks[best_params[1]])
print ('The best model was trained with regularization parameter %s' % regParams[best_params[0]])
print ('The best model was trained with rank %s' % ranks[best_params[1]])
my_model = models[best_params[0]][best_params[1]]


# In[ ]:


#Example of predicted plays
predicted_counts_df.show(10)


# In[ ]:


#test = test.withColumn("counts", test["counts"].cast(DoubleType()))
predict_df = my_model.transform(test)


# In[ ]:


predicted_test_df = predict_df.filter(predict_df.prediction != float('nan'))


# In[ ]:


predicted_test_df = predicted_test_df.withColumn("prediction", F.abs(F.round(predicted_test_df["prediction"],0)))
# Run the previously created RMSE evaluator, reg_eval, on the predicted_test_df DataFrame
test_RMSE = reg_eval.evaluate(predicted_test_df)


# In[ ]:


print('The model had a RMSE on the test set of {0}'.format(test_RMSE))


# # Baseline Method

# In[ ]:


avg_pref_df = sqlContext.sql("select avg('rating') from traing")
avg_pref_df.show()


# In[ ]:


Extract the average preference value. (This is row 0, column 0.)
training_avg_pref= avg_pref_df.collect()[0][0]
training_avg_pref


# In[ ]:


print('The average number of preference in the dataset is {0}'.format(training_avg_pref))


# In[ ]:


# Add a column with the average preference
test_for_avg_df = test.withColumn('prediction',F.lit(4.0))


# In[ ]:


# Run the previously created RMSE evaluator, reg_eval, on the test_for_avg_df DataFrame
test_avg_RMSE = reg_eval.evaluate(test_for_avg_df)


# In[ ]:


print("The RMSE on the average set is {0}".format(test_avg_RMSE))


# ## Prediction

# In[ ]:


def recommendMovies(model, user, nbRecommendations):

    # Create a Spark DataFrame with the specified user and all the movies listed in the ratings DataFrame

    dataSet = data.select("product_id").distinct().withColumn("user_id", F.lit(user))

    # Create a Spark DataFrame with the movies that have already been rated by this user

    moviesAlreadyRated = data.filter(data.user_id == user).select("product_id", "user_id")

    # Apply the recommender system to the data set without the already rated movies to predict ratings

    predictions = model.transform(dataSet.subtract(moviesAlreadyRated)).dropna().orderBy("prediction", ascending= False).limit(nbRecommendations).select("user_id",'product_id',"prediction")

    # Join with the movies DataFrame to get the movies titles and genres

    recommendations = predictions.join(products, ['product_id']).select('user_id','product_name')
    

    return recommendations


# In[ ]:


user_id_list = data.select("user_id").distinct()


# In[ ]:


user_list=user_id_list.rdd.map(lambda x: x.user_id).collect()


# In[ ]:


df=recommendMovies(my_model,496,5)
for user_id in user_list:
    df=df.append(recommendMovies(my_model,user_id,5))  


# In[ ]:


df.write.format("text").option("header", "false").mode("append").save("output.txt")


# In[ ]:


df.toPandas().to_csv('mycsv.csv')


# In[ ]:


df=recommendMovies(my_model,224,5)

df1=recommendMovies(my_model,151,5)

df2=recommendMovies(my_model,210,5)

df3=recommendMovies(my_model,89,5)

dfn1=df.union(df1)
dfn2=dfn1.union(df2)
dfn3=dfn2.union(df3)
dfn3.show()
#df.show()


# In[ ]:


df.to_csv('pred-677.csv', sep=',', encoding='utf-8')

