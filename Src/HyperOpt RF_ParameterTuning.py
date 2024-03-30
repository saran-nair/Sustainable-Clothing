# Databricks notebook source
textile_df = spark.read.csv("dbfs:/FileStore/shared_uploads/saran@uni-koblenz.de/textile.csv", header="true", inferSchema="true", multiLine="true", escape='"')
#display(textile_df)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import expr
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# COMMAND ----------

DF1  = textile_df.drop("Alpaca", "Other_animal","Camel","Other_regenerated","Other_plant","Jute")

# COMMAND ----------

DF_1 = DF1

DF_1 = DF_1.withColumn("Manufacturing_location_Africa", when(DF_1["Manufacturing_location"] == "Africa", 1).otherwise(0))
DF_1 = DF_1.withColumn("Manufacturing_location_America", when(DF_1["Manufacturing_location"] == "America", 1).otherwise(0))
DF_1 = DF_1.withColumn("Manufacturing_location_Asia", when(DF_1["Manufacturing_location"] == "Asia", 1).otherwise(0))
DF_1 = DF_1.withColumn("Manufacturing_location_Europe", when(DF_1["Manufacturing_location"] == "Europe", 1).otherwise(0))
DF_1 = DF_1.withColumn("Manufacturing_location_Unknown", when((DF_1["Manufacturing_location"].isNull()) | (DF_1["Manufacturing_location"] == "NaN"), 1).otherwise(0))

DF_1 = DF_1.withColumn("Drying_instruction_Linedry", when(DF_1["Drying_instruction"] == "Line dry", 1).otherwise(0))
DF_1 = DF_1.withColumn("Drying_instruction_Dryclean", when(DF_1["Drying_instruction"] == "Dry clean", 1).otherwise(0))
DF_1 = DF_1.withColumn("Drying_instruction_Tumble", when((DF_1["Drying_instruction"] == "Tumble dry_ low") | (DF_1["Drying_instruction"] == "Tumble dry_ low"),1).otherwise(0))


DF_1 = DF_1.withColumn("Washing_instruction_Machinehot", when((DF_1["Washing_instruction"] == "Machine wash_ warm") | (DF_1["Washing_instruction"] == "Machine wash_ hot"),1).otherwise(0))
DF_1 = DF_1.withColumn("Washing_instruction_Machinecold", when(DF_1["Washing_instruction"] == "Machine wash_ cold",1).otherwise(0))
DF_1 = DF_1.withColumn("Washing_instruction_Handwash", when(DF_1["Washing_instruction"] == "Hand wash",1).otherwise(0))
DF_1 = DF_1.withColumn("Washing_instruction_Dryclean", when(DF_1["Washing_instruction"] == "Dry clean",1).otherwise(0))

DF_1 = DF_1.drop("Drying_instruction")
DF_1 = DF_1.drop("Manufacturing_location")
DF_1 = DF_1.drop("Washing_instruction")

# COMMAND ----------

train_df, test_df = DF_1.randomSplit([.8, .2], seed=42)
categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") | (dataType == "integer") & (field != "EI"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

#Hyper-Parameter Tuning
num_trees = 50
max_depth = 15
max_bins = 30
feature_subset_strategy = "all"
subsampling_rate = 0.8
min_instances_per_node = 2
min_info_gain = 0.01

rf_classifier = RandomForestClassifier(labelCol="EI", 
                                       numTrees=num_trees, 
                                       maxDepth=max_depth, 
                                       maxBins=max_bins, 
                                       featureSubsetStrategy=feature_subset_strategy, 
                                       subsamplingRate=subsampling_rate, 
                                       minInstancesPerNode=min_instances_per_node, 
                                       minInfoGain=min_info_gain)

# Update the stages for the pipeline
stages = [string_indexer, vec_assembler, rf_classifier]

# Create a pipeline
pipeline = Pipeline(stages=stages)

# Fit the pipeline on the training data
pipeline_model = pipeline.fit(train_df)

# Make predictions on the test data
pred_df = pipeline_model.transform(test_df)

evaluator = MulticlassClassificationEvaluator(labelCol="EI", predictionCol="prediction")

# Calculate accuracy
accuracy = evaluator.evaluate(pred_df, {evaluator.metricName: "accuracy"})

# Calculate precision
precision = evaluator.evaluate(pred_df, {evaluator.metricName: "weightedPrecision"})

# Calculate recall
recall = evaluator.evaluate(pred_df, {evaluator.metricName: "weightedRecall"})

# Calculate F1 score
f1_score = evaluator.evaluate(pred_df, {evaluator.metricName: "f1"})

# Print the evaluation metrics
print("Accuracy on test data = {:.2%}".format(accuracy))
print("Precision = {:.2%}".format(precision))
print("Recall = {:.2%}".format(recall))
print("F1 Score = {:.2%}".format(f1_score))

# COMMAND ----------

from hyperopt import fmin, tpe, hp, Trials

# COMMAND ----------

space = {
    'numTrees': hp.quniform('numTrees', 10, 100, 1),
    'maxDepth': hp.quniform('maxDepth', 5, 20, 1),
    'maxBins': hp.quniform('maxBins', 23, 50, 1),
    'featureSubsetStrategy': hp.choice('featureSubsetStrategy', ['auto', 'all', 'sqrt', 'log2']),
    'subsamplingRate': hp.uniform('subsamplingRate', 0.5, 1.0),
    'minInstancesPerNode': hp.quniform('minInstancesPerNode', 1, 10, 1),
    'minInfoGain': hp.uniform('minInfoGain', 0.0, 0.1)
}

# COMMAND ----------

def objective(params):
    # Convert hyperopt params to int where necessary
    params['numTrees'] = int(params['numTrees'])
    params['maxDepth'] = int(params['maxDepth'])
    params['maxBins'] = int(params['maxBins'])
    params['minInstancesPerNode'] = int(params['minInstancesPerNode'])
    
    # Configure Random Forest Classifier with hyperparameters
    rf_classifier = RandomForestClassifier(labelCol="EI", **params)

    # Update the stages for the pipeline
    stages = [string_indexer, vec_assembler, rf_classifier]

    # Create a pipeline
    pipeline = Pipeline(stages=stages)

    # Fit the pipeline on the training data
    pipeline_model = pipeline.fit(train_df)

    # Make predictions on the test data
    pred_df = pipeline_model.transform(test_df)

    # Evaluate predictions
    evaluator = MulticlassClassificationEvaluator(labelCol="EI", predictionCol="prediction")
    accuracy = evaluator.evaluate(pred_df, {evaluator.metricName: "accuracy"})
    return -accuracy  # Minimize negative accuracy (maximize accuracy)

# Use Trials for single-machine optimization
trials = Trials()

# Run Hyperopt optimization
best_params = fmin(fn=objective,
                   space=space,
                   algo=tpe.suggest,
                   max_evals=50,
                   trials=trials)

# Print the best parameters found
print("Best parameters:", best_params)

# COMMAND ----------

# Convert best_params to int where necessary
best_params['numTrees'] = int(best_params['numTrees'])
best_params['maxDepth'] = int(best_params['maxDepth'])
best_params['maxBins'] = int(best_params['maxBins'])
best_params['minInstancesPerNode'] = int(best_params['minInstancesPerNode'])
best_params['featureSubsetStrategy'] = "all"

# Train the final model using the best parameters
rf_classifier_best = RandomForestClassifier(labelCol="EI", **best_params)
stages_best = [string_indexer, vec_assembler, rf_classifier_best]
pipeline_best = Pipeline(stages=stages_best)
pipeline_model_best = pipeline_best.fit(train_df)
pred_df_best = pipeline_model_best.transform(test_df)

# Evaluate the final model
accuracy_best = evaluator.evaluate(pred_df_best, {evaluator.metricName: "accuracy"})
precision_best = evaluator.evaluate(pred_df_best, {evaluator.metricName: "weightedPrecision"})
recall_best = evaluator.evaluate(pred_df_best, {evaluator.metricName: "weightedRecall"})
f1_score_best = evaluator.evaluate(pred_df_best, {evaluator.metricName: "f1"})

# Print evaluation metrics for the best model
print("Best Model Evaluation Metrics:")
print("Accuracy on test data = {:.2%}".format(accuracy_best))
print("Precision = {:.2%}".format(precision_best))
print("Recall = {:.2%}".format(recall_best))
print("F1 Score = {:.2%}".format(f1_score_best))

# COMMAND ----------


