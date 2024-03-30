# Databricks notebook source
textile_df = spark.read.csv("dbfs:/FileStore/shared_uploads/saran@uni-koblenz.de/textile.csv", header="true", inferSchema="true", multiLine="true", escape='"')
display(textile_df)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.ml.classification import GBTClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import expr
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# COMMAND ----------

DF1  = textile_df.drop("Alpaca", "Other_animal","Camel","Other_regenerated","Other_plant","Jute")
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

# MAGIC %md
# MAGIC MODEL

# COMMAND ----------

train_df, test_df = DF_1.randomSplit([.8, .2], seed=42)

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") | (dataType == "integer") & (field != "EI"))]
assembler_inputs = index_output_cols + numeric_cols

# Create a GBT classifier
gbt = GBTClassifier(featuresCol="features", 
                    labelCol="EI", 
                    maxDepth=20, 
                    maxIter=15, 
                    maxBins=30,
                    subsamplingRate=0.8,
                    stepSize=0.2,
                    minInstancesPerNode=1,
                    minInfoGain=0.01,
                    featureSubsetStrategy="all")

# Create a OneVsRest classifier
ovr_classifier = OneVsRest(classifier=gbt, labelCol="EI")

# Create a VectorAssembler to combine categorical and numeric features
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# Build the pipeline
pipeline_gbt = Pipeline(stages=[string_indexer, vec_assembler, ovr_classifier])

# Train the GBT model
model_gbt = pipeline_gbt.fit(train_df)

# Make predictions on the test data
pred_df_gbt = model_gbt.transform(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC EVALUATION

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(labelCol="EI", predictionCol="prediction")

accuracy = evaluator.evaluate(pred_df_gbt, {evaluator.metricName: "accuracy"})

precision = evaluator.evaluate(pred_df_gbt, {evaluator.metricName: "weightedPrecision"})

recall = evaluator.evaluate(pred_df_gbt, {evaluator.metricName: "weightedRecall"})

f1_score = evaluator.evaluate(pred_df_gbt, {evaluator.metricName: "f1"})

labels = pred_df_gbt.select('EI').rdd.map(lambda row: row[0])
labels_list = labels.collect()

# Initialize lists to store fpr and tpr for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute ROC curve and ROC area for each class
for label in labels_list:
    # Filter predictions for the current label
    pred_and_label = pred_df_gbt.select(['prediction', 'EI']).rdd.map(lambda row: (float(row['prediction'] == label), 1.0 if float(row['EI']) == label else 0.0))

    fpr[label], tpr[label], _ = roc_curve(pred_and_label.map(lambda x: x[1]).collect(), pred_and_label.map(lambda x: x[0]).collect())
    
    # Compute AUC
    roc_auc[label] = auc(fpr[label], tpr[label])

average_auc = sum(roc_auc.values()) / len(roc_auc)
# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for label in labels_list:
    plt.plot(fpr[label], tpr[label])

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right", title=f'Average AUC = {average_auc:.2f}')
plt.show()

print("Accuracy on test data = {:.2%}".format(accuracy))
print("Precision = {:.2%}".format(precision))
print("Recall = {:.2%}".format(recall))
print("F1 Score = {:.2%}".format(f1_score))

# COMMAND ----------

# MAGIC %md
# MAGIC RECOMMENDATIONS

# COMMAND ----------

full_predictions = model_gbt.transform(DF_1)
sustainable_products = full_predictions.filter(full_predictions["prediction"].isin([1, 2]))
recommended_items = sustainable_products.orderBy(col("prediction").asc())
recommended_items.select("ID", "Type", "prediction", "EI").show()

# COMMAND ----------

filtered_jackets = recommended_items.filter(recommended_items["Type"] == "jacket")
filtered_jackets.select("ID", "Type", "EI").show()

# COMMAND ----------

filtered_blouse = recommended_items.filter(recommended_items["Type"] == "blouse")
filtered_blouse.select("ID", "Type", "EI").show()

# COMMAND ----------

filtered_shirt = recommended_items.filter(recommended_items["Type"] == "shirt")
filtered_shirt.select("ID", "Type", "EI").show()

# COMMAND ----------


