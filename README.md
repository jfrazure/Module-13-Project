# Module-13-Project

Jonathan Frazure

Venture Funding with Deep Learning

You work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked you to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

The business team has given you a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. With your knowledge of machine learning and neural networks, you decide to use the features in the provided dataset to create a binary classifier model that will predict whether an applicant will become a successful business. The CSV file contains a variety of information about these businesses, including whether or not they ultimately became successful.

Instructions:

The steps for this challenge are broken out into the following sections:

Prepare the data for use on a neural network model.

Compile and evaluate a binary classification model using a neural network.

Optimize the neural network model.

Prepare the Data for Use on a Neural Network Model

Using your knowledge of Pandas and scikit-learn’s StandardScaler(), preprocess the dataset so that you can use it to compile and evaluate the neural network model later.

Open the starter code file, and complete the following data preparation steps:

Read the applicants_data.csv file into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define your features and target variables.

Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they are not relevant to the binary classification model.

Encode the dataset’s categorical variables using OneHotEncoder, and then place the encoded variables into a new DataFrame.

Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.

Note To complete this step, you will employ the Pandas concat() function that was introduced earlier in this course.

Using the preprocessed data, create the features (X) and target (y) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns should define the features dataset.

Split the features and target sets into training and testing datasets.

Use scikit-learn's StandardScaler to scale the features data.

Compile and Evaluate a Binary Classification Model Using a Neural Network

Use your knowledge of TensorFlow to design a binary classification deep neural network model. This model should use the dataset’s features to predict whether an Alphabet Soup–funded startup will be successful based on the features in the dataset. Consider the number of inputs before determining the number of layers that your model will contain or the number of neurons on each layer. Then, compile and fit your model. Finally, evaluate your binary classification model to calculate the model’s loss and accuracy.

To do so, complete the following steps:

Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.
Hint You can start with a two-layer deep neural network model that uses the relu activation function for both layers.

Compile and fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric.
Hint When fitting the model, start with a small number of epochs, such as 20, 50, or 100.

Evaluate the model using the test data to determine the model’s loss and accuracy.

Save and export your model to an HDF5 file, and name the file AlphabetSoup.h5.

Optimize the Neural Network Model

Using your knowledge of TensorFlow and Keras, optimize your model to improve the model's accuracy. Even if you do not successfully achieve a better accuracy, you'll need to demonstrate at least two attempts to optimize the model. You can include these attempts in your existing notebook. Or, you can make copies of the starter notebook in the same folder, rename them, and code each model optimization in a new notebook.

Note You will not lose points if your model does not achieve a high accuracy, as long as you make at least two attempts to optimize the model.

To do so, complete the following steps:

Define at least three new deep neural network models (the original plus 2 optimization attempts). With each, try to improve on your first model’s predictive accuracy.
Rewind Recall that perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize your model for a predictive accuracy as close to 1 as possible, you can use any or all of the following techniques:

Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.

Add more neurons (nodes) to a hidden layer.

Add more hidden layers.

Use different activation functions for the hidden layers.

Add to or reduce the number of epochs in the training regimen.

After finishing your models, display the accuracy scores achieved by each model, and compare the results.

Save each of your models as an HDF5 file.

# Imports
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
Prepare the data to be used on a neural network model
Step 1: Read the applicants_data.csv file into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define your features and target variables.
# Read the applicants_data.csv file from the Resources folder into a Pandas DataFrame
applicant_data_df = pd.read_csv(Path('./Resources/applicants_data.csv'))

# Review the DataFrame
applicant_data_df
EIN	NAME	APPLICATION_TYPE	AFFILIATION	CLASSIFICATION	USE_CASE	ORGANIZATION	STATUS	INCOME_AMT	SPECIAL_CONSIDERATIONS	ASK_AMT	IS_SUCCESSFUL
0	10520599	BLUE KNIGHTS MOTORCYCLE CLUB	T10	Independent	C1000	ProductDev	Association	1	0	N	5000	1
1	10531628	AMERICAN CHESAPEAKE CLUB CHARITABLE TR	T3	Independent	C2000	Preservation	Co-operative	1	1-9999	N	108590	1
2	10547893	ST CLOUD PROFESSIONAL FIREFIGHTERS	T5	CompanySponsored	C3000	ProductDev	Association	1	0	N	5000	0
3	10553066	SOUTHSIDE ATHLETIC ASSOCIATION	T3	CompanySponsored	C2000	Preservation	Trust	1	10000-24999	N	6692	1
4	10556103	GENETIC RESEARCH INSTITUTE OF THE DESERT	T3	Independent	C1000	Heathcare	Trust	1	100000-499999	N	142590	1
...	...	...	...	...	...	...	...	...	...	...	...	...
34294	996009318	THE LIONS CLUB OF HONOLULU KAMEHAMEHA	T4	Independent	C1000	ProductDev	Association	1	0	N	5000	0
34295	996010315	INTERNATIONAL ASSOCIATION OF LIONS CLUBS	T4	CompanySponsored	C3000	ProductDev	Association	1	0	N	5000	0
34296	996012607	PTA HAWAII CONGRESS	T3	CompanySponsored	C2000	Preservation	Association	1	0	N	5000	0
34297	996015768	AMERICAN FEDERATION OF GOVERNMENT EMPLOYEES LO...	T5	Independent	C3000	ProductDev	Association	1	0	N	5000	1
34298	996086871	WATERHOUSE CHARITABLE TR	T3	Independent	C1000	Preservation	Co-operative	1	1M-5M	N	36500179	0
34299 rows × 12 columns

# Review the data types associated with the columns
applicant_data_df.dtypes
EIN                        int64
NAME                      object
APPLICATION_TYPE          object
AFFILIATION               object
CLASSIFICATION            object
USE_CASE                  object
ORGANIZATION              object
STATUS                     int64
INCOME_AMT                object
SPECIAL_CONSIDERATIONS    object
ASK_AMT                    int64
IS_SUCCESSFUL              int64
dtype: object
Step 2: Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they are not relevant to the binary classification model.
# Drop the 'EIN' and 'NAME' columns from the DataFrame
applicant_data_df = applicant_data_df.drop(columns=['EIN', 'NAME'])

# Review the DataFrame
applicant_data_df
APPLICATION_TYPE	AFFILIATION	CLASSIFICATION	USE_CASE	ORGANIZATION	STATUS	INCOME_AMT	SPECIAL_CONSIDERATIONS	ASK_AMT	IS_SUCCESSFUL
0	T10	Independent	C1000	ProductDev	Association	1	0	N	5000	1
1	T3	Independent	C2000	Preservation	Co-operative	1	1-9999	N	108590	1
2	T5	CompanySponsored	C3000	ProductDev	Association	1	0	N	5000	0
3	T3	CompanySponsored	C2000	Preservation	Trust	1	10000-24999	N	6692	1
4	T3	Independent	C1000	Heathcare	Trust	1	100000-499999	N	142590	1
...	...	...	...	...	...	...	...	...	...	...
34294	T4	Independent	C1000	ProductDev	Association	1	0	N	5000	0
34295	T4	CompanySponsored	C3000	ProductDev	Association	1	0	N	5000	0
34296	T3	CompanySponsored	C2000	Preservation	Association	1	0	N	5000	0
34297	T5	Independent	C3000	ProductDev	Association	1	0	N	5000	1
34298	T3	Independent	C1000	Preservation	Co-operative	1	1M-5M	N	36500179	0
34299 rows × 10 columns

Step 3: Encode the dataset’s categorical variables using OneHotEncoder, and then place the encoded variables into a new DataFrame.
# Create a list of categorical variables 
categorical_variables = list(applicant_data_df.dtypes[applicant_data_df.dtypes == "object"].index)

# Display the categorical variables list
categorical_variables
['APPLICATION_TYPE',
 'AFFILIATION',
 'CLASSIFICATION',
 'USE_CASE',
 'ORGANIZATION',
 'INCOME_AMT',
 'SPECIAL_CONSIDERATIONS']
# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse_output=False)
# Encode the categorcal variables using OneHotEncoder
encoded_data = enc.fit_transform(applicant_data_df[categorical_variables])
# Create a DataFrame with the encoded variables
encoded_df = pd.DataFrame(
    encoded_data,
    columns = enc.get_feature_names_out(categorical_variables)
)

# Review the DataFrame
encoded_df
APPLICATION_TYPE_T10	APPLICATION_TYPE_T12	APPLICATION_TYPE_T13	APPLICATION_TYPE_T14	APPLICATION_TYPE_T15	APPLICATION_TYPE_T17	APPLICATION_TYPE_T19	APPLICATION_TYPE_T2	APPLICATION_TYPE_T25	APPLICATION_TYPE_T29	...	INCOME_AMT_1-9999	INCOME_AMT_10000-24999	INCOME_AMT_100000-499999	INCOME_AMT_10M-50M	INCOME_AMT_1M-5M	INCOME_AMT_25000-99999	INCOME_AMT_50M+	INCOME_AMT_5M-10M	SPECIAL_CONSIDERATIONS_N	SPECIAL_CONSIDERATIONS_Y
0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
1	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
2	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
3	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
4	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
34294	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
34295	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
34296	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
34297	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
34298	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	1.0	0.0
34299 rows × 114 columns

Step 4: Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.

Note To complete this step, you will employ the Pandas concat() function that was introduced earlier in this course.

# Add the numerical variables from the original DataFrame to the one-hot encoding DataFrame
encoded_df = pd.concat(
    [
        encoded_df,
        applicant_data_df[list(applicant_data_df.dtypes[applicant_data_df.dtypes == "int64"].index)]
    ],
    axis=1
)

# Review the Dataframe
encoded_df
APPLICATION_TYPE_T10	APPLICATION_TYPE_T12	APPLICATION_TYPE_T13	APPLICATION_TYPE_T14	APPLICATION_TYPE_T15	APPLICATION_TYPE_T17	APPLICATION_TYPE_T19	APPLICATION_TYPE_T2	APPLICATION_TYPE_T25	APPLICATION_TYPE_T29	...	INCOME_AMT_10M-50M	INCOME_AMT_1M-5M	INCOME_AMT_25000-99999	INCOME_AMT_50M+	INCOME_AMT_5M-10M	SPECIAL_CONSIDERATIONS_N	SPECIAL_CONSIDERATIONS_Y	STATUS	ASK_AMT	IS_SUCCESSFUL
0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	5000	1
1	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	108590	1
2	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	5000	0
3	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	6692	1
4	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	142590	1
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
34294	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	5000	0
34295	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	5000	0
34296	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	5000	0
34297	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	5000	1
34298	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	1.0	0.0	0.0	0.0	1.0	0.0	1	36500179	0
34299 rows × 117 columns

Step 5: Using the preprocessed data, create the features (X) and target (y) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns should define the features dataset.
# Define the target set y using the IS_SUCCESSFUL column
y = encoded_df['IS_SUCCESSFUL']

# Display a sample of y
y
0        1
1        1
2        0
3        1
4        1
        ..
34294    0
34295    0
34296    0
34297    1
34298    0
Name: IS_SUCCESSFUL, Length: 34299, dtype: int64
# Define features set X by selecting all columns but IS_SUCCESSFUL
X = encoded_df.drop(columns='IS_SUCCESSFUL')

# Review the features DataFrame
X
APPLICATION_TYPE_T10	APPLICATION_TYPE_T12	APPLICATION_TYPE_T13	APPLICATION_TYPE_T14	APPLICATION_TYPE_T15	APPLICATION_TYPE_T17	APPLICATION_TYPE_T19	APPLICATION_TYPE_T2	APPLICATION_TYPE_T25	APPLICATION_TYPE_T29	...	INCOME_AMT_100000-499999	INCOME_AMT_10M-50M	INCOME_AMT_1M-5M	INCOME_AMT_25000-99999	INCOME_AMT_50M+	INCOME_AMT_5M-10M	SPECIAL_CONSIDERATIONS_N	SPECIAL_CONSIDERATIONS_Y	STATUS	ASK_AMT
0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	5000
1	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	108590
2	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	5000
3	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	6692
4	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	1.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	142590
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
34294	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	5000
34295	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	5000
34296	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	5000
34297	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	1	5000
34298	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	1.0	0.0	0.0	0.0	1.0	0.0	1	36500179
34299 rows × 116 columns

Step 6: Split the features and target sets into training and testing datasets.
# Split the preprocessed data into a training and testing dataset
# Assign the function a random_state equal to 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
Step 7: Use scikit-learn's StandardScaler to scale the features data.
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the features training dataset
X_scaler = scaler.fit(X_train)

# Fit the scaler to the features training dataset
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
Compile and Evaluate a Binary Classification Model Using a Neural Network
Step 1: Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.

Hint You can start with a two-layer deep neural network model that uses the relu activation function for both layers.

# Define the the number of inputs (features) to the model
number_input_features = len(X_train.iloc[0])

# Review the number of features
number_input_features
116
# Define the number of neurons in the output layer
number_output_neurons = 1
# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1 =  (number_input_features + 1) // 2

# Review the number hidden nodes in the first layer
hidden_nodes_layer1
58
# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer2 =  (hidden_nodes_layer1 + 1) // 2

# Review the number hidden nodes in the second layer
hidden_nodes_layer2
29
# Create the Sequential model instance
nn = Sequential()
# Add the first hidden layer
nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))
# Add the second hidden layer
nn.add(Dense(units=hidden_nodes_layer2, activation="relu"))
# Add the output layer to the model specifying the number of output neurons and activation function
nn.add(Dense(units=1, activation="sigmoid"))
# Display the Sequential model summary
nn.summary()
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 58)                6786      
                                                                 
 dense_1 (Dense)             (None, 29)                1711      
                                                                 
 dense_2 (Dense)             (None, 1)                 30        
                                                                 
=================================================================
Total params: 8527 (33.31 KB)
Trainable params: 8527 (33.31 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Step 2: Compile and fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric.
# Compile the Sequential model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# Fit the model using 50 epochs and the training data
fit_model = nn.fit(X_train_scaled, y_train, epochs=50)
Epoch 1/50
804/804 [==============================] - 1s 363us/step - loss: 0.5770 - accuracy: 0.7229
Epoch 2/50
804/804 [==============================] - 0s 357us/step - loss: 0.5541 - accuracy: 0.7301
Epoch 3/50
804/804 [==============================] - 0s 351us/step - loss: 0.5500 - accuracy: 0.7315
Epoch 4/50
804/804 [==============================] - 0s 354us/step - loss: 0.5475 - accuracy: 0.7310
Epoch 5/50
804/804 [==============================] - 0s 355us/step - loss: 0.5457 - accuracy: 0.7313
Epoch 6/50
804/804 [==============================] - 0s 353us/step - loss: 0.5441 - accuracy: 0.7330
Epoch 7/50
804/804 [==============================] - 0s 349us/step - loss: 0.5439 - accuracy: 0.7332
Epoch 8/50
804/804 [==============================] - 0s 356us/step - loss: 0.5428 - accuracy: 0.7331
Epoch 9/50
804/804 [==============================] - 0s 352us/step - loss: 0.5419 - accuracy: 0.7352
Epoch 10/50
804/804 [==============================] - 0s 353us/step - loss: 0.5412 - accuracy: 0.7356
Epoch 11/50
804/804 [==============================] - 0s 347us/step - loss: 0.5414 - accuracy: 0.7366
Epoch 12/50
804/804 [==============================] - 0s 341us/step - loss: 0.5403 - accuracy: 0.7360
Epoch 13/50
804/804 [==============================] - 0s 352us/step - loss: 0.5397 - accuracy: 0.7369
Epoch 14/50
804/804 [==============================] - 0s 360us/step - loss: 0.5393 - accuracy: 0.7377
Epoch 15/50
804/804 [==============================] - 0s 549us/step - loss: 0.5391 - accuracy: 0.7372
Epoch 16/50
804/804 [==============================] - 0s 367us/step - loss: 0.5386 - accuracy: 0.7372
Epoch 17/50
804/804 [==============================] - 0s 342us/step - loss: 0.5389 - accuracy: 0.7379
Epoch 18/50
804/804 [==============================] - 0s 344us/step - loss: 0.5377 - accuracy: 0.7380
Epoch 19/50
804/804 [==============================] - 0s 342us/step - loss: 0.5383 - accuracy: 0.7367
Epoch 20/50
804/804 [==============================] - 0s 347us/step - loss: 0.5376 - accuracy: 0.7381
Epoch 21/50
804/804 [==============================] - 0s 344us/step - loss: 0.5372 - accuracy: 0.7368
Epoch 22/50
804/804 [==============================] - 0s 341us/step - loss: 0.5369 - accuracy: 0.7385
Epoch 23/50
804/804 [==============================] - 0s 343us/step - loss: 0.5367 - accuracy: 0.7390
Epoch 24/50
804/804 [==============================] - 0s 370us/step - loss: 0.5360 - accuracy: 0.7392
Epoch 25/50
804/804 [==============================] - 0s 358us/step - loss: 0.5361 - accuracy: 0.7389
Epoch 26/50
804/804 [==============================] - 0s 356us/step - loss: 0.5361 - accuracy: 0.7390
Epoch 27/50
804/804 [==============================] - 0s 343us/step - loss: 0.5356 - accuracy: 0.7381
Epoch 28/50
804/804 [==============================] - 0s 345us/step - loss: 0.5353 - accuracy: 0.7391
Epoch 29/50
804/804 [==============================] - 0s 347us/step - loss: 0.5352 - accuracy: 0.7390
Epoch 30/50
804/804 [==============================] - 0s 344us/step - loss: 0.5349 - accuracy: 0.7393
Epoch 31/50
804/804 [==============================] - 0s 349us/step - loss: 0.5346 - accuracy: 0.7399
Epoch 32/50
804/804 [==============================] - 0s 349us/step - loss: 0.5344 - accuracy: 0.7393
Epoch 33/50
804/804 [==============================] - 0s 356us/step - loss: 0.5344 - accuracy: 0.7397
Epoch 34/50
804/804 [==============================] - 0s 359us/step - loss: 0.5340 - accuracy: 0.7396
Epoch 35/50
804/804 [==============================] - 0s 363us/step - loss: 0.5338 - accuracy: 0.7409
Epoch 36/50
804/804 [==============================] - 0s 359us/step - loss: 0.5336 - accuracy: 0.7403
Epoch 37/50
804/804 [==============================] - 0s 349us/step - loss: 0.5332 - accuracy: 0.7409
Epoch 38/50
804/804 [==============================] - 0s 349us/step - loss: 0.5332 - accuracy: 0.7403
Epoch 39/50
804/804 [==============================] - 0s 370us/step - loss: 0.5332 - accuracy: 0.7398
Epoch 40/50
804/804 [==============================] - 0s 363us/step - loss: 0.5328 - accuracy: 0.7404
Epoch 41/50
804/804 [==============================] - 0s 362us/step - loss: 0.5326 - accuracy: 0.7403
Epoch 42/50
804/804 [==============================] - 0s 353us/step - loss: 0.5327 - accuracy: 0.7410
Epoch 43/50
804/804 [==============================] - 0s 361us/step - loss: 0.5326 - accuracy: 0.7405
Epoch 44/50
804/804 [==============================] - 0s 367us/step - loss: 0.5321 - accuracy: 0.7416
Epoch 45/50
804/804 [==============================] - 0s 348us/step - loss: 0.5323 - accuracy: 0.7408
Epoch 46/50
804/804 [==============================] - 0s 347us/step - loss: 0.5321 - accuracy: 0.7411
Epoch 47/50
804/804 [==============================] - 0s 349us/step - loss: 0.5321 - accuracy: 0.7409
Epoch 48/50
804/804 [==============================] - 0s 349us/step - loss: 0.5316 - accuracy: 0.7407
Epoch 49/50
804/804 [==============================] - 0s 349us/step - loss: 0.5314 - accuracy: 0.7414
Epoch 50/50
804/804 [==============================] - 0s 351us/step - loss: 0.5317 - accuracy: 0.7414
Step 3: Evaluate the model using the test data to determine the model’s loss and accuracy.
# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
268/268 - 0s - loss: 0.5590 - accuracy: 0.7276 - 114ms/epoch - 426us/step
Loss: 0.5590173006057739, Accuracy: 0.727580189704895
Step 4: Save and export your model to an HDF5 file, and name the file AlphabetSoup.h5.
# Set the model's file path
file_path = "./Resources/AlphabetSoup.h5"

# Export your model to a HDF5 file
nn.save_weights(file_path)
Optimize the neural network model
Step 1: Define at least three new deep neural network models (resulting in the original plus 3 optimization attempts). With each, try to improve on your first model’s predictive accuracy.

Rewind Recall that perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize your model for a predictive accuracy as close to 1 as possible, you can use any or all of the following techniques:

Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.

Add more neurons (nodes) to a hidden layer.

Add more hidden layers.

Use different activation functions for the hidden layers.

Add to or reduce the number of epochs in the training regimen.

Alternative Model 1
# Define the the number of inputs (features) to the model
number_input_features = len(X_train.iloc[0])

# Review the number of features
number_input_features
116
# Define the number of neurons in the output layer
number_output_neurons_A1 = 1
# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1_A1 = (number_input_features + 1) // 2

# Review the number of hidden nodes in the first layer
hidden_nodes_layer1_A1
58
# Additional code JF added for multiple hiddent layers to satisfy above requirements

# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer2_A1 = (hidden_nodes_layer1_A1 + 1) // 2

# Review the number hidden nodes in the second layer
hidden_nodes_layer2_A1
29
# Additional code JF added for multiple hiddent layers to satisfy above requirements

# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer3_A1 = (hidden_nodes_layer2_A1 + 1) // 2

# Review the number hidden nodes in the second layer
hidden_nodes_layer3_A1
15
# Create the Sequential model instance
nn_A1 = Sequential()
# First hidden layer
nn_A1.add(Dense(units=hidden_nodes_layer1_A1, input_dim=number_input_features, activation="relu"))

# Add the second hidden layer
nn_A1.add(Dense(units=hidden_nodes_layer2_A1, activation="relu"))

# Add the third hidden layer
nn_A1.add(Dense(units=hidden_nodes_layer3_A1, activation="relu"))

# Output layer
nn_A1.add(Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn_A1.summary()
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_3 (Dense)             (None, 58)                6786      
                                                                 
 dense_4 (Dense)             (None, 29)                1711      
                                                                 
 dense_5 (Dense)             (None, 15)                450       
                                                                 
 dense_6 (Dense)             (None, 1)                 16        
                                                                 
=================================================================
Total params: 8963 (35.01 KB)
Trainable params: 8963 (35.01 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
# Compile the Sequential model
nn_A1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# Fit the model using 50 epochs and the training data
fit_model_A1 = nn_A1.fit(X_train_scaled, y_train, epochs=50)
Epoch 1/50
804/804 [==============================] - 0s 395us/step - loss: 0.5748 - accuracy: 0.7183
Epoch 2/50
804/804 [==============================] - 0s 395us/step - loss: 0.5528 - accuracy: 0.7310
Epoch 3/50
804/804 [==============================] - 0s 393us/step - loss: 0.5488 - accuracy: 0.7307
Epoch 4/50
804/804 [==============================] - 0s 398us/step - loss: 0.5475 - accuracy: 0.7324
Epoch 5/50
804/804 [==============================] - 0s 395us/step - loss: 0.5453 - accuracy: 0.7336
Epoch 6/50
804/804 [==============================] - 0s 390us/step - loss: 0.5446 - accuracy: 0.7338
Epoch 7/50
804/804 [==============================] - 0s 398us/step - loss: 0.5434 - accuracy: 0.7336
Epoch 8/50
804/804 [==============================] - 0s 409us/step - loss: 0.5420 - accuracy: 0.7356
Epoch 9/50
804/804 [==============================] - 0s 403us/step - loss: 0.5422 - accuracy: 0.7348
Epoch 10/50
804/804 [==============================] - 0s 387us/step - loss: 0.5417 - accuracy: 0.7347
Epoch 11/50
804/804 [==============================] - 0s 397us/step - loss: 0.5410 - accuracy: 0.7355
Epoch 12/50
804/804 [==============================] - 0s 396us/step - loss: 0.5405 - accuracy: 0.7350
Epoch 13/50
804/804 [==============================] - 0s 399us/step - loss: 0.5398 - accuracy: 0.7367
Epoch 14/50
804/804 [==============================] - 0s 390us/step - loss: 0.5392 - accuracy: 0.7372
Epoch 15/50
804/804 [==============================] - 0s 392us/step - loss: 0.5391 - accuracy: 0.7376
Epoch 16/50
804/804 [==============================] - 0s 390us/step - loss: 0.5383 - accuracy: 0.7377
Epoch 17/50
804/804 [==============================] - 0s 388us/step - loss: 0.5382 - accuracy: 0.7376
Epoch 18/50
804/804 [==============================] - 0s 383us/step - loss: 0.5378 - accuracy: 0.7377
Epoch 19/50
804/804 [==============================] - 0s 386us/step - loss: 0.5369 - accuracy: 0.7389
Epoch 20/50
804/804 [==============================] - 0s 385us/step - loss: 0.5371 - accuracy: 0.7391
Epoch 21/50
804/804 [==============================] - 0s 392us/step - loss: 0.5367 - accuracy: 0.7383
Epoch 22/50
804/804 [==============================] - 0s 390us/step - loss: 0.5364 - accuracy: 0.7396
Epoch 23/50
804/804 [==============================] - 0s 391us/step - loss: 0.5363 - accuracy: 0.7386
Epoch 24/50
804/804 [==============================] - 0s 448us/step - loss: 0.5360 - accuracy: 0.7392
Epoch 25/50
804/804 [==============================] - 0s 404us/step - loss: 0.5360 - accuracy: 0.7399
Epoch 26/50
804/804 [==============================] - 0s 398us/step - loss: 0.5357 - accuracy: 0.7387
Epoch 27/50
804/804 [==============================] - 0s 408us/step - loss: 0.5358 - accuracy: 0.7390
Epoch 28/50
804/804 [==============================] - 0s 408us/step - loss: 0.5353 - accuracy: 0.7393
Epoch 29/50
804/804 [==============================] - 0s 401us/step - loss: 0.5348 - accuracy: 0.7406
Epoch 30/50
804/804 [==============================] - 0s 400us/step - loss: 0.5348 - accuracy: 0.7407
Epoch 31/50
804/804 [==============================] - 0s 390us/step - loss: 0.5349 - accuracy: 0.7393
Epoch 32/50
804/804 [==============================] - 0s 413us/step - loss: 0.5343 - accuracy: 0.7386
Epoch 33/50
804/804 [==============================] - 0s 410us/step - loss: 0.5343 - accuracy: 0.7400
Epoch 34/50
804/804 [==============================] - 0s 408us/step - loss: 0.5342 - accuracy: 0.7395
Epoch 35/50
804/804 [==============================] - 0s 404us/step - loss: 0.5337 - accuracy: 0.7402
Epoch 36/50
804/804 [==============================] - 0s 406us/step - loss: 0.5341 - accuracy: 0.7404
Epoch 37/50
804/804 [==============================] - 0s 405us/step - loss: 0.5338 - accuracy: 0.7402
Epoch 38/50
804/804 [==============================] - 0s 403us/step - loss: 0.5337 - accuracy: 0.7410
Epoch 39/50
804/804 [==============================] - 0s 391us/step - loss: 0.5330 - accuracy: 0.7406
Epoch 40/50
804/804 [==============================] - 0s 389us/step - loss: 0.5330 - accuracy: 0.7406
Epoch 41/50
804/804 [==============================] - 0s 388us/step - loss: 0.5324 - accuracy: 0.7411
Epoch 42/50
804/804 [==============================] - 0s 409us/step - loss: 0.5327 - accuracy: 0.7406
Epoch 43/50
804/804 [==============================] - 0s 413us/step - loss: 0.5328 - accuracy: 0.7407
Epoch 44/50
804/804 [==============================] - 0s 418us/step - loss: 0.5325 - accuracy: 0.7402
Epoch 45/50
804/804 [==============================] - 0s 411us/step - loss: 0.5323 - accuracy: 0.7411
Epoch 46/50
804/804 [==============================] - 0s 413us/step - loss: 0.5325 - accuracy: 0.7405
Epoch 47/50
804/804 [==============================] - 0s 414us/step - loss: 0.5320 - accuracy: 0.7410
Epoch 48/50
804/804 [==============================] - 0s 400us/step - loss: 0.5321 - accuracy: 0.7416
Epoch 49/50
804/804 [==============================] - 0s 389us/step - loss: 0.5318 - accuracy: 0.7413
Epoch 50/50
804/804 [==============================] - 0s 389us/step - loss: 0.5318 - accuracy: 0.7409
Alternative Model 2
# Define the the number of inputs (features) to the model
number_input_features = len(X_train.iloc[0])

# Review the number of features
number_input_features
116
# Define the number of neurons in the output layer
number_output_neurons_A2 = 1
# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1_A2 = (number_input_features + 1) // 2

# Review the number of hidden nodes in the first layer
hidden_nodes_layer1_A2
58
# Create the Sequential model instance
nn_A2 = Sequential()
# First hidden layer
nn_A2.add(Dense(input_dim=number_input_features, units=hidden_nodes_layer1_A2, activation="relu"))

# Output layer
nn_A2.add(Dense(units=number_output_neurons_A2, activation="sigmoid"))

# Check the structure of the model
nn_A2.summary()
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_8 (Dense)             (None, 58)                6786      
                                                                 
 dense_9 (Dense)             (None, 1)                 59        
                                                                 
=================================================================
Total params: 6845 (26.74 KB)
Trainable params: 6845 (26.74 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
# Compile the model
nn_A2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# Fit the model
nn_A2.fit(X_train_scaled, y_train, epochs=50)
Epoch 1/50
804/804 [==============================] - 0s 326us/step - loss: 0.5810 - accuracy: 0.7197
Epoch 2/50
804/804 [==============================] - 0s 325us/step - loss: 0.5584 - accuracy: 0.7273
Epoch 3/50
804/804 [==============================] - 0s 320us/step - loss: 0.5533 - accuracy: 0.7300
Epoch 4/50
804/804 [==============================] - 0s 320us/step - loss: 0.5520 - accuracy: 0.7311
Epoch 5/50
804/804 [==============================] - 0s 319us/step - loss: 0.5499 - accuracy: 0.7317
Epoch 6/50
804/804 [==============================] - 0s 326us/step - loss: 0.5495 - accuracy: 0.7314
Epoch 7/50
804/804 [==============================] - 0s 336us/step - loss: 0.5480 - accuracy: 0.7315
Epoch 8/50
804/804 [==============================] - 0s 320us/step - loss: 0.5474 - accuracy: 0.7322
Epoch 9/50
804/804 [==============================] - 0s 312us/step - loss: 0.5473 - accuracy: 0.7318
Epoch 10/50
804/804 [==============================] - 0s 310us/step - loss: 0.5461 - accuracy: 0.7335
Epoch 11/50
804/804 [==============================] - 0s 310us/step - loss: 0.5464 - accuracy: 0.7324
Epoch 12/50
804/804 [==============================] - 0s 312us/step - loss: 0.5454 - accuracy: 0.7327
Epoch 13/50
804/804 [==============================] - 0s 307us/step - loss: 0.5447 - accuracy: 0.7340
Epoch 14/50
804/804 [==============================] - 0s 308us/step - loss: 0.5452 - accuracy: 0.7327
Epoch 15/50
804/804 [==============================] - 0s 310us/step - loss: 0.5445 - accuracy: 0.7336
Epoch 16/50
804/804 [==============================] - 0s 307us/step - loss: 0.5439 - accuracy: 0.7335
Epoch 17/50
804/804 [==============================] - 0s 309us/step - loss: 0.5437 - accuracy: 0.7343
Epoch 18/50
804/804 [==============================] - 0s 321us/step - loss: 0.5433 - accuracy: 0.7348
Epoch 19/50
804/804 [==============================] - 0s 311us/step - loss: 0.5432 - accuracy: 0.7347
Epoch 20/50
804/804 [==============================] - 0s 308us/step - loss: 0.5428 - accuracy: 0.7346
Epoch 21/50
804/804 [==============================] - 0s 308us/step - loss: 0.5423 - accuracy: 0.7350
Epoch 22/50
804/804 [==============================] - 0s 314us/step - loss: 0.5427 - accuracy: 0.7346
Epoch 23/50
804/804 [==============================] - 0s 309us/step - loss: 0.5419 - accuracy: 0.7341
Epoch 24/50
804/804 [==============================] - 0s 318us/step - loss: 0.5416 - accuracy: 0.7349
Epoch 25/50
804/804 [==============================] - 0s 310us/step - loss: 0.5411 - accuracy: 0.7359
Epoch 26/50
804/804 [==============================] - 0s 308us/step - loss: 0.5415 - accuracy: 0.7358
Epoch 27/50
804/804 [==============================] - 0s 308us/step - loss: 0.5412 - accuracy: 0.7350
Epoch 28/50
804/804 [==============================] - 0s 320us/step - loss: 0.5415 - accuracy: 0.7354
Epoch 29/50
804/804 [==============================] - 0s 332us/step - loss: 0.5409 - accuracy: 0.7360
Epoch 30/50
804/804 [==============================] - 0s 330us/step - loss: 0.5405 - accuracy: 0.7361
Epoch 31/50
804/804 [==============================] - 0s 338us/step - loss: 0.5412 - accuracy: 0.7358
Epoch 32/50
804/804 [==============================] - 0s 319us/step - loss: 0.5401 - accuracy: 0.7363
Epoch 33/50
804/804 [==============================] - 0s 316us/step - loss: 0.5409 - accuracy: 0.7350
Epoch 34/50
804/804 [==============================] - 0s 327us/step - loss: 0.5402 - accuracy: 0.7366
Epoch 35/50
804/804 [==============================] - 0s 322us/step - loss: 0.5393 - accuracy: 0.7361
Epoch 36/50
804/804 [==============================] - 0s 325us/step - loss: 0.5397 - accuracy: 0.7364
Epoch 37/50
804/804 [==============================] - 0s 328us/step - loss: 0.5403 - accuracy: 0.7364
Epoch 38/50
804/804 [==============================] - 0s 318us/step - loss: 0.5398 - accuracy: 0.7359
Epoch 39/50
804/804 [==============================] - 0s 311us/step - loss: 0.5397 - accuracy: 0.7355
Epoch 40/50
804/804 [==============================] - 0s 320us/step - loss: 0.5396 - accuracy: 0.7357
Epoch 41/50
804/804 [==============================] - 0s 334us/step - loss: 0.5390 - accuracy: 0.7383
Epoch 42/50
804/804 [==============================] - 0s 333us/step - loss: 0.5393 - accuracy: 0.7378
Epoch 43/50
804/804 [==============================] - 0s 324us/step - loss: 0.5390 - accuracy: 0.7361
Epoch 44/50
804/804 [==============================] - 0s 325us/step - loss: 0.5389 - accuracy: 0.7378
Epoch 45/50
804/804 [==============================] - 0s 318us/step - loss: 0.5390 - accuracy: 0.7364
Epoch 46/50
804/804 [==============================] - 0s 318us/step - loss: 0.5387 - accuracy: 0.7359
Epoch 47/50
804/804 [==============================] - 0s 311us/step - loss: 0.5392 - accuracy: 0.7360
Epoch 48/50
804/804 [==============================] - 0s 308us/step - loss: 0.5387 - accuracy: 0.7357
Epoch 49/50
804/804 [==============================] - 0s 313us/step - loss: 0.5388 - accuracy: 0.7361
Epoch 50/50
804/804 [==============================] - 0s 312us/step - loss: 0.5389 - accuracy: 0.7366
<keras.src.callbacks.History at 0x29a25ab60>
Step 2: After finishing your models, display the accuracy scores achieved by each model, and compare the results.
print("Original Model Results")

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
Original Model Results
268/268 - 0s - loss: 0.5590 - accuracy: 0.7276 - 77ms/epoch - 288us/step
Loss: 0.5590173006057739, Accuracy: 0.727580189704895
print("Alternative Model 1 Results")

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn_A1.evaluate(X_test_scaled,y_test,verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
Alternative Model 1 Results
268/268 - 0s - loss: 0.5543 - accuracy: 0.7322 - 109ms/epoch - 408us/step
Loss: 0.5542652010917664, Accuracy: 0.7322449088096619
print("Alternative Model 2 Results")

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn_A2.evaluate(X_test_scaled,y_test,verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
Alternative Model 2 Results
268/268 - 0s - loss: 0.5607 - accuracy: 0.7325 - 103ms/epoch - 386us/step
Loss: 0.5606561303138733, Accuracy: 0.732478141784668
Step 3: Save each of your alternative models as an HDF5 file.
# Set the file path for the first alternative model
file_path_A1 = ("Resources/AlphabetSoupA1.h5")


# Export your model to a HDF5 file
nn_A1.save_weights(file_path_A1)
# Set the file path for the second alternative model
file_path_A2 = ("Resources/AlphabetSoupA2.h5")

# Export your model to a HDF5 file
nn_A2.save_weights(file_path_A2)
Click to add a cell.

Simple
0
20
Python 3 (ipykernel) | Idle
