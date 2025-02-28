# -*- coding: utf-8 -*-
"""
Created on march 17 09:40:27 2024

@author: Rezaul Hasan Bhuiyan; student id: s3003337
"""

#Importing necessary packages
import rasterio
from rasterio.plot import show
import geopandas as gpd
import fiona
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
import os

## Setting up working directory 
BASE_DIR = "C:/Users/User/Desktop/Python_solution/Dataset"
os.chdir(BASE_DIR)

#input files
ndviStack = 'NDVI_12.tif'
samplingPoints = 'samplingPoints.shp'

# output_files
temp_point_data = 'temp_Y_point'

# Crop type names & band names for post visualization
cropType_name = ['Cereals','Lucerne', 'Maize', 'Onions', 'Orchard','Potatoes', 'Sugar Beet']
band_names = ['November', 'May', 'February', 
              'April', 'January','September', 
              'July', 'August', 'June', 
              'March', 'December', 'October']  # Each bands represents the calculated NDVI of the months of the year". 

##Inspect the data
#raster layer
ndvi = rasterio.open(ndviStack)
ndvi.meta

#vector layer
Points = gpd.read_file(samplingPoints)
Points.shape
Points.crs
# # if the projection system is not matching try this
# with rasterio.open(ndviStack) as ndvi_file:
#     # Read the metadata
#     ndvi_meta = ndvi_file.meta
#     # Update the CRS (Coordinate Reference System) information
#     new_crs = 'EPSG:4326'  # Example
#     ndvi_meta['crs'] = new_crs
# # Update the metadata
# ndvi_meta.update()

# Count the occurrences of each crop type
crop_type_counts = Points['Class'].value_counts()

# Plot the histogram
plt.figure(figsize=(10, 6))
crop_type_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Samples for Each Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize the sampling points and NDVI layer in a same figure to understant sampling distribution
fig, ax = plt.subplots(figsize = (12,12))
Points.plot(ax= ax, color = 'orangered')
show(ndvi, ax=ax, cmap = 'viridis')

## Feature Extraction for RF model
# reading ndvi bands from input
with rasterio.open(ndviStack) as img:
    bands = img.count
# Using iteration to assign band names from the band_names list
features = [band_names[i] for i in range(bands)]
print('Bands names: ', features)
f_len = len(features)
# Read sampling points data, assign IDs, and save to a temporary point file
points = gpd.read_file(samplingPoints)
# adding a new column 'id' with range of points
points = points.assign(id=range(len(points)))
# saving new point file with 'id'
points.to_file(temp_point_data) 

# Converting gdf to pd df and removing geometry
points_df = pd.DataFrame(points.drop(columns='geometry'))
# ilterating over multiband raster
sampled = pd.Series()
# Read input shapefile with fiona and iterate over each feature
with fiona.open(temp_point_data) as shp:
    for feature in shp:
        #print(feature)
        siteID = feature['properties']['id']
        coords = feature['geometry']['coordinates']
        # Read pixel value at the given coordinates using Rasterio
        # NB: `sample()` returns an iterable of ndarrays.
        with rasterio.open(ndviStack) as stack_src:
                  value = [v for v in stack_src.sample([coords])]
        # Update the pandas serie accordingly
        sampled.loc[siteID] = value
        
# Reshape sampled values into DataFrame
df1 = pd.DataFrame(sampled.values.tolist(), index=sampled.index)
df1['id'] = df1.index
df1 = pd.DataFrame(df1[0].values.tolist(), columns=features)
df1['id'] = df1.index

# Merging dataset for trainig model
data = pd.merge(df1, points_df, on ='id')
print('Sampled Data: \n',data)

## Split data into training and testing sets
x = data.iloc[:,0:f_len]
X = x.values
y = data.iloc[:,-1]
Y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, stratify = Y)
print(f'X_train Shape: {X_train.shape}\nX_test Shape: {X_test.shape}\ny_train Shape: {y_train.shape}\ny_test Shape:{y_test.shape}')


## Training Model
cName = 'RF' #classifier Name

##Compare hyperparameter combinations to find the best settings using OOB error and accuracy.
# Initialize lists to store results
accuracies = []
oob_errors = []

# Define different values for hyperparameters
n_trees = [100, 500, 1000]
mtry = [2, 5, 10]

# Iterate over different hyperparameter values
for n_tree in n_trees:
    for m in mtry:
        # Initialize Random Forest classifier with specified hyperparameters
        clf = RandomForestClassifier(n_estimators=n_tree, max_features=m, random_state=42, oob_score=True)
        clf.fit(X_train, y_train)
        
        # Evaluate classifier performance on testing data
        clf_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, clf_pred)
        accuracies.append(accuracy)
        
        # Get OOB error
        oob_error = 1 - clf.oob_score_
        oob_errors.append(oob_error)

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(len(accuracies)), accuracies, marker='o')
plt.xticks(range(len(accuracies)), [f"({n},{m})" for n in n_tree for m in mtry])
plt.xlabel('Hyperparameter Combination')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Hyperparameters')
plt.grid(True)
plt.show()

# Plot OOB error
plt.figure(figsize=(10, 6))
plt.plot(range(len(oob_errors)), oob_errors, marker='o', color='orange')
plt.xticks(range(len(oob_errors)), [f"({n},{m})" for n in n_tree for m in mtry])
plt.xlabel('Hyperparameter Combination')
plt.ylabel('OOB Error')
plt.title('OOB Error vs Hyperparameters')
plt.grid(True)
plt.show()

# Set hyperparameters to best combination for final prediction
ntree = 500  # Number of trees
mtry =  2    # Number of variables used for splitting

# Initialize Random Forest classifier with specified hyperparameters
clf = RandomForestClassifier(n_estimators=ntree, max_features=mtry, random_state=42)
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)

## Accuracy Assessment
print(f"Accuracy {cName}: {accuracy_score(y_test, clf_pred)*100}")
print(classification_report(y_test, clf_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, clf_pred)
# Calculate percentages for each class
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plotting confusion matrix 
plt.figure(figsize=(12, 8))
sns.set(font_scale=1.5)

# Iterate through the confusion matrix percentages and annotate the heatmap with formatted strings
for i in range(len(cm_percent)):
    for j in range(len(cm_percent[i])):
        plt.text(j+0.5, i+0.5, f"{cm_percent[i, j]:.1f}", 
                 ha='center', va='center', color='black')

sns.heatmap(cm_percent, annot= False, cmap='Blues', 
            xticklabels=cropType_name, yticklabels=cropType_name, cbar=True)
plt.title('Confusion Matrix', fontsize = 'large')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate producer accuracy (PA) and user accuracy (UA) for each class
producer_accuracy = np.diag(cm) / np.sum(cm, axis=1)
user_accuracy = np.diag(cm) / np.sum(cm, axis=0)

# Calculate overall producer and user accuracy
overall_producer_accuracy = np.mean(producer_accuracy)
overall_user_accuracy = np.mean(user_accuracy)

# Calculate Kappa coefficient
kappa_coefficient = cohen_kappa_score(y_test, clf_pred)

# Print the results
print("Producer Accuracy (PA):", producer_accuracy)
print("User Accuracy (UA):", user_accuracy)
print("Overall Producer Accuracy:", overall_producer_accuracy)
print("Overall User Accuracy:", overall_user_accuracy)
print("Kappa Coefficient:", kappa_coefficient)

##Get feature importances
feature_importances = clf.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(feature_importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
sorted_features = [features[i] for i in indices]
# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances[indices], y=np.array(sorted_features))
plt.title("Feature Importance Hierarchy")
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()

##full data reshaping, predicting Croptypes for whole image, and saving output
exp_name = f'Croptype_{cName}.tif'
img = rasterio.open(ndviStack)
img_arr = img.read()
img_arr = np.where(np.isnan(img_arr),0,img_arr) #replacing nana valuse with zeros
bands = img_arr.shape[0]
print(f'Height: {img_arr.shape[1]}\nWidth: {img_arr.shape[2]}\nBands: {img_arr.shape[0]}\n')
img_n = np.moveaxis(img_arr, 0, -1)
img_n.shape
img_n = img_n.reshape(-1, f_len)
print('reshaped full data shape  for prediction: ',img_n.shape)

# predict Croptypes for whole area
pred_CropTypes = clf.predict(img_n)

# Define colors for each crop type
colors = {
    'Cereals': '#FFC107',     # Yellow
    'Lucerne': '#4CAF50',     # Green
    'Maize': '#FF5722',       # Orange
    'Onions': '#795548',      # Brown
    'Orchard': '#673AB7',     # Purple
    'Potatoes': '#607D8B',    # Gray
    'Sugar Beet': '#E91E63'   # Pink
}

# Create a legend patch for each crop type
legend_patches = [plt.Rectangle((0,0),1,1,fc=colors[c], edgecolor='none') for c in cropType_name]

# Plot the classified image
plt.figure(figsize=(18, 16))
plt.imshow(pred_CropTypes.reshape(img_arr.shape[1], img_arr.shape[2]), cmap='tab10',interpolation='nearest')
plt.title('Map of Crop Types', fontsize='large')
plt.legend(handles=legend_patches, 
           labels=cropType_name, loc='upper left', 
           bbox_to_anchor=(1.0, 1), 
           title='Crop Types',
           fontsize='large')
plt.axis('off')  # Remove axis
plt.show()

## Predefining out raster meta using variable raster
tempfile_arr = img.read(1)
tempfile_arr = tempfile_arr.reshape(-1,1)
metadata = img.meta

height = metadata.get('height')
width = metadata.get('width')
crs = metadata.get('crs')
transform = metadata.get('transform')

img_reshape = pred_CropTypes.reshape(height, width)

# Write predicted values to output raster
out_raster = rasterio.open(exp_name,
                                         'w',
                                          driver='GTiff',
                                          height=height,
                                          width=width,
                                          count=1,
                                          dtype='uint8',
                                          crs=crs,
                                          transform = transform,
                                          nodata = 255 #nodata
                                          )

out_raster.write(img_reshape, 1)
out_raster.shape

## Close all opened files
img.close()
ndvi.close()
out_raster.close()
