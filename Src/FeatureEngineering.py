# Databricks notebook source
textile_df = spark.read.csv("dbfs:/FileStore/shared_uploads/saran@uni-koblenz.de/textile.csv", header="true", inferSchema="true", multiLine="true", escape='"')
display(textile_df)

# COMMAND ----------

duplicates = textile_df.groupBy(textile_df.columns).count().where('count > 1').select(textile_df.columns).dropDuplicates()
#duplicates.show()
display(duplicates)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in ['Type', 'Manufacturing_location', 'Use_location', 'Washing_instruction', 'Drying_instruction']]
encoder = OneHotEncoder(inputCols=[col+"_index" for col in ['Type', 'Manufacturing_location', 'Use_location', 'Washing_instruction', 'Drying_instruction']], outputCols=[col+"_encoded" for col in ['Type', 'Manufacturing_location', 'Use_location', 'Washing_instruction', 'Drying_instruction']])

# Assemble features into a vector
assembler = VectorAssembler(inputCols=[col+"_encoded" for col in ['Type', 'Manufacturing_location', 'Use_location', 'Washing_instruction', 'Drying_instruction']] + [other_column for other_column in textile_df.columns if other_column not in ['Type', 'Manufacturing_location', 'Use_location', 'Washing_instruction', 'Drying_instruction', 'EI']], outputCol="features")

# Initialize ChiSqSelector
selector = ChiSqSelector(numTopFeatures=15, featuresCol="features", outputCol="selectedFeatures", labelCol="EI")

# Define pipeline
pipeline = Pipeline(stages=indexers + [encoder, assembler, selector])

# Fit pipeline model
model = pipeline.fit(textile_df)

# Transform data
result = model.transform(textile_df)

# Get selected features' column names
selected_features_columns = result.schema["selectedFeatures"].metadata["ml_attr"]["attrs"]["numeric"][:15]

# Print selected features
print("Selected features:")
for feature in selected_features_columns:
    print(feature["name"])


# COMMAND ----------

display(textile_df.describe())

# COMMAND ----------

display(textile_df.summary())

# COMMAND ----------

display(textile_df.summary())

# COMMAND ----------

dbutils.data.summarize(textile_df)

# COMMAND ----------

display(textile_df.select("Washing_instruction").distinct())

# COMMAND ----------

display(textile_df.select("Drying_instruction").distinct())

# COMMAND ----------

display(textile_df.select("Type").distinct())

# COMMAND ----------

display(textile_df.select("Manufacturing_location").distinct())

# COMMAND ----------

textile_df.filter(col("Manufacturing_location")== 'NaN').count()

# COMMAND ----------

display(textile_df.groupBy("Manufacturing_location").count().orderBy(col("Manufacturing_location")))

# COMMAND ----------

display(textile_df.select("Use_location").distinct())

# COMMAND ----------

from pyspark.sql.functions import col
display(textile_df.groupBy("EI").count().orderBy(col("EI")))

# COMMAND ----------

numerical_cols = [col_name for col_name, data_type in textile_df.dtypes if data_type != 'string'and col_name != 'ID']
for col1 in numerical_cols:
    for col2 in numerical_cols:
        if col1 != col2:
            correlation_value = textile_df.stat.corr(col1, col2, method='pearson')
            print(f"Pearson's correlation between {col1} and {col2}: {correlation_value}")

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

numerical_cols = [col_name for col_name, data_type in textile_df.dtypes if data_type != 'string'and col_name != 'ID']
correlation_matrix = pd.DataFrame(index=numerical_cols, columns=numerical_cols)

for col1 in numerical_cols:
    for col2 in numerical_cols:
        if col1 != col2:
            correlation_value = textile_df.stat.corr(col1, col2, method='pearson')
            correlation_matrix.loc[col1, col2] = correlation_value

# Convert the correlation matrix to a DataFrame
correlation_matrix = correlation_matrix.astype(float)

# Visualize the heatmap
plt.figure(figsize=(16, 14))
plt.title('Correlation Heatmap')
plt.xlabel('Variables')
plt.ylabel('Variables')
plt.imshow(correlation_matrix, cmap='inferno', interpolation='nearest')
plt.colorbar()
plt.xticks(ticks=range(len(numerical_cols)), labels=numerical_cols, rotation=90)
plt.yticks(ticks=range(len(numerical_cols)), labels=numerical_cols)
plt.show()


# COMMAND ----------

DF1  = textile_df.drop("Alpaca", "Other_animal","Camel","Other_regenerated","Other_plant","Jute")

# COMMAND ----------

numerical_cols1 = [col_name for col_name, data_type in DF1.dtypes if data_type != 'string'and col_name != 'ID']
correlation_matrix1 = pd.DataFrame(index=numerical_cols1, columns=numerical_cols1)

for col1 in numerical_cols1:
    for col2 in numerical_cols1:
        if col1 != col2:
            correlation_value1 = DF1.stat.corr(col1, col2, method='pearson')
            correlation_matrix1.loc[col1, col2] = correlation_value1

# Convert the correlation matrix to a DataFrame
correlation_matrix1 = correlation_matrix1.astype(float)

# Visualize the heatmap
plt.figure(figsize=(16,13))
plt.title('Correlation Heatmap')
plt.xlabel('Variables')
plt.ylabel('Variables')
plt.imshow(correlation_matrix1, cmap='inferno', interpolation='nearest')
plt.colorbar()
plt.xticks(ticks=range(len(numerical_cols1)), labels=numerical_cols1, rotation=90)
plt.yticks(ticks=range(len(numerical_cols1)), labels=numerical_cols1)
plt.show()

# COMMAND ----------

import numpy as np

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, PCA, Normalizer
from pyspark.ml.linalg import Vectors

numerical_columns = [col_name for col_name, data_type in DF1.dtypes if data_type != 'string'and col_name != 'ID']

indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in ['Type', 'Manufacturing_location', 'Use_location', 'Washing_instruction', 'Drying_instruction']]

# Assemble features into a vector
assembler = VectorAssembler(inputCols=[col+"_index" for col in ['Type', 'Manufacturing_location', 'Use_location', 'Washing_instruction', 'Drying_instruction']] + numerical_columns, outputCol="features")

# Apply indexers to the DataFrame
indexed_df = DF1
for indexer in indexers:
    indexed_df = indexer.fit(indexed_df).transform(indexed_df)

assembled_df = assembler.transform(indexed_df).select("features")

#Scale the features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
scaled_df = scaler.fit(assembled_df).transform(assembled_df)

#Perform LDA by PCA
lda = PCA(k=36, inputCol="scaled_features", outputCol="lda_features")
lda_model = lda.fit(scaled_df)
transformed_df = lda_model.transform(scaled_df)

transformed_df.show()

# COMMAND ----------

transformed_df.printSchema()

transformed_df.select("features").schema[0].metadata

# COMMAND ----------

feature_names = ['Cotton', 'Organic_cotton', 'Linen', 'Hemp', 'Silk', 'Wool', 'Leather', 'Cashmere', 'Feathers', 'Polyester', 'Nylon', 'Acrylic', 'Spandex', 'Elastane', 'Polyamide', 'Other_synthetic', 'Lyocell', 'Viscose', 'Acetate', 'Modal', 'Rayon', 'Other', 'Recycled_content', 'Reused_content', 'Material_label', 'Chemicals_label', 'Production_label', 'Transporation_distance', 'Reusability_label', 'Recylability_label','EI','Type_index', 'Manufacturing_location_index', 'Use_location_index', 'Washing_instruction_index', 'Drying_instruction_index']

# Access the explained variance ratio of each principal component
explained_variance_ratio = lda_model.explainedVariance

# Access the principal component loadings
principal_component_loadings = lda_model.pc.toArray()

# Print the explained variance ratio of each principal component
print("Explained Variance Ratio:")
for i, explained_variance in enumerate(explained_variance_ratio):
    print(f"Principal Component {i + 1}: {explained_variance:.4f}")

# Print the loadings of each original feature for the first few principal components
print("Principal Component Loadings:")
for i, component_loadings in enumerate(principal_component_loadings[:5]):  # Print loadings for first few components
    print(f"Principal Component {i + 1}:")
    for j, loading in enumerate(component_loadings):
        feature_name = feature_names[j]
        print(f"  {feature_name}: {loading:.4f}")

# Decide the number of components to keep based on the cumulative explained variance ratio
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
num_components_to_keep = np.argmax(cumulative_explained_variance_ratio >= 0.95) + 1
print(f"Number of components to keep: {num_components_to_keep}")


# COMMAND ----------

from pyspark.sql.functions import when
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

display(DF_1.groupBy("Manufacturing_location_Africa").count().orderBy(col("Manufacturing_location_Africa")))

# COMMAND ----------

display(DF_1.groupBy("Manufacturing_location_Unknown").count().orderBy(col("Manufacturing_location_Unknown")))

# COMMAND ----------

numerical_cols1 = [col_name for col_name, data_type in DF_1.dtypes if data_type != 'string'and col_name != 'ID']
correlation_matrix1 = pd.DataFrame(index=numerical_cols1, columns=numerical_cols1)

for col1 in numerical_cols1:
    for col2 in numerical_cols1:
        if col1 != col2:
            correlation_value1 = DF_1.stat.corr(col1, col2, method='pearson')
            correlation_matrix1.loc[col1, col2] = correlation_value1

# Convert the correlation matrix to a DataFrame
correlation_matrix1 = correlation_matrix1.astype(float)

# Visualize the heatmap
plt.figure(figsize=(16,13))
plt.title('Correlation Heatmap')
plt.xlabel('Variables')
plt.ylabel('Variables')
plt.imshow(correlation_matrix1, cmap='inferno', interpolation='nearest')
plt.colorbar()
plt.xticks(ticks=range(len(numerical_cols1)), labels=numerical_cols1, rotation=90)
plt.yticks(ticks=range(len(numerical_cols1)), labels=numerical_cols1)
plt.show()

# COMMAND ----------

quant_columns = [col_name for col_name, data_type in DF_1.dtypes if data_type != 'string'and col_name != 'ID']

indexers1 = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in ['Type','Use_location']]

# Assemble features into a vector
assembler1 = VectorAssembler(inputCols=[col+"_index" for col in ['Type','Use_location']] + quant_columns, outputCol="features")

# Apply indexers to the DataFrame
indexed_df1 = DF_1
for indexer in indexers1:
    indexed_df1 = indexer.fit(indexed_df1).transform(indexed_df1)

assembled_df1 = assembler1.transform(indexed_df1).select("features")

#Scale the features
scaler1 = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
scaled_df1 = scaler1.fit(assembled_df1).transform(assembled_df1)

#Perform LDA by PCA
lda1 = PCA(k=45, inputCol="scaled_features", outputCol="lda_features")
lda_model1 = lda1.fit(scaled_df1)
transformed_df1 = lda_model1.transform(scaled_df1)


feature_names = ['Cotton', 'Organic_cotton', 'Linen', 'Hemp', 'Silk', 'Wool', 'Leather', 'Cashmere', 'Feathers', 'Polyester', 'Nylon', 'Acrylic', 'Spandex', 'Elastane', 'Polyamide', 'Other_synthetic', 'Lyocell', 'Viscose', 'Acetate', 'Modal', 'Rayon', 'Other', 'Recycled_content', 'Reused_content', 'Material_label', 'Chemicals_label', 'Production_label', 'Transporation_distance', 'Reusability_label', 'Recylability_label','EI','Type_index', 'Manufacturing_location_Africa', 'Manufacturing_location_America','Manufacturing_location_Asia','Manufacturing_location_Europe','Manufacturing_location_Unknown', 'Use_location_index', 'Washing_instruction_Machinehot','Washing_instruction_Machinecold','Washing_instruction_Handwash','Washing_instruction_Dryclean', 'Drying_instruction_Linedry','Drying_instruction_Dryclean','Drying_instruction_Tumble']

# Access the explained variance ratio of each principal component
explained_variance_ratio = lda_model1.explainedVariance

# Access the principal component loadings
principal_component_loadings = lda_model1.pc.toArray()

# Print the explained variance ratio of each principal component
print("Explained Variance Ratio:")
for i, explained_variance in enumerate(explained_variance_ratio):
    print(f"Principal Component {i + 1}: {explained_variance:.4f}")

# Print the loadings of each original feature for the first few principal components
print("Principal Component Loadings:")
for i, component_loadings in enumerate(principal_component_loadings[:5]): 
    print(f"Principal Component {i + 1}:")
    for j, loading in enumerate(component_loadings):
        feature_name = feature_names[j]
        print(f"  {feature_name}: {loading:.4f}")

# Decide the number of components to keep based on the cumulative explained variance ratio
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
num_components_to_keep = np.argmax(cumulative_explained_variance_ratio >= 0.95) + 1
print(f"Number of components to keep: {num_components_to_keep}")

# COMMAND ----------

DF_2 = DF_1.drop("Linen","Hemp","Silk","Wool","Leather","Cashmere","Feathers","Nylon","Acrylic","Spandex","Polyamide","Other_synthetic","Viscose","Acetate","Modal","Rayon","Other","Recycled_content")
DF_3 = DF_1.drop("Drying_instruction_Linedry","Drying_instruction_Dryclean","Drying_instruction_Tumble")

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col

train_df, test_df = DF_3.randomSplit([.8,.2],seed=42)

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") | (dataType == "integer") & (field != "EI"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

max_depth = 20
min_instances_per_node = 1
min_info_gain = 0.01
impurity = 'entropy'

dt_classifier = DecisionTreeClassifier(maxDepth=max_depth, minInstancesPerNode=min_instances_per_node, minInfoGain=min_info_gain, impurity=impurity,labelCol="EI")

stages = [string_indexer, vec_assembler, dt_classifier]

pipeline = Pipeline(stages=stages)

pipeline_model = pipeline.fit(train_df)

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

#from pyspark.ml.classification import RandomForestClassifier
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

print("Test Error = %g" % (1.0 - accuracy))

# Get all unique labels in the dataset
# Get all unique labels in the dataset
labels = [float(row.EI) for row in test_df.select('EI').distinct().collect()]

# Initialize lists to store fpr and tpr for each label
fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute ROC curve and ROC area for each class
for label in labels:
    # Filter predictions for the current label
    pred_and_label = pred_df.select(['probability', 'EI']).rdd.map(lambda row: (float(row['probability'][int(label)]), 1.0 if float(row['EI']) == label else 0.0))

    # Use sklearn's roc_curve to compute fpr and tpr
    fpr[label], tpr[label], _ = roc_curve(pred_and_label.map(lambda x: x[1]).collect(), pred_and_label.map(lambda x: x[0]).collect())
    
    # Compute AUC
    roc_auc[label] = auc(fpr[label], tpr[label])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for label in labels:
    plt.plot(fpr[label], tpr[label], label='ROC curve of "EI" {0} (area = {1:0.2f})'.format(label, roc_auc[label]))

plt.plot([0, 1], [0, 1], 'k--')  # random curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve [One V/s Rest Strategy]')
plt.legend(loc="lower right")
plt.show()

# Compute overall AUC
mean_auc = np.mean(list(roc_auc.values()))
print("Mean AUC:", mean_auc)

# COMMAND ----------

dt_model = pipeline_model.stages[-1]
display(dt_model)

# COMMAND ----------

vector_assembler = pipeline_model.stages[1] 
feature_names = vector_assembler.getInputCols()

# Get the DecisionTree model
dt_model = pipeline_model.stages[-1]

# Retrieve feature importances from the DecisionTree model
feature_importances = dt_model.featureImportances

# Combine feature names and importances into a dictionary
feature_importance_dict = dict(zip(feature_names, feature_importances))

# Display feature importances
for feature, importance in feature_importance_dict.items():
    print(f"{feature}: {importance}")


# COMMAND ----------

import pandas as pd

features_df = pd.DataFrame(list(zip(vector_assembler.getInputCols(), dt_model.featureImportances)), columns=["Feature", "Importance"])
features_df_sorted = features_df.sort_values(by="Importance", ascending=False)
features_df_sorted.head(20)

# COMMAND ----------


