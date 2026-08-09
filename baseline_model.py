from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import data_processing as dt

# all protein interactions in the file
all_interaction = dt.read_score_file(r"C:\Users\sujng\OneDrive\바탕_화면\Columbia_Spring_2026\Intermediate_Python\HW\final-project-sk5755\4932.protein.links.v12.0.txt")
# only interactions with high scores
valid_interaction = dt.valid_interactions(all_interaction)
# protein id to amino acid seqeunce
protein_id = dt.read_protein_id(r"C:\Users\sujng\OneDrive\바탕_화면\Columbia_Spring_2026\Intermediate_Python\HW\final-project-sk5755\4932.protein.sequences.v12.0.fa")
# protein id to features
features = dt.id_to_feature(protein_id)   
# non-interaction protein pairs
neg_data = dt.negative_set(protein_id, all_interaction)
features_array_pos, weight_array_pos = dt.array_conversion(valid_interaction, features)
features_array_neg, weight_array_neg = dt.array_conversion(neg_data,features)

X = np.vstack((features_array_pos, features_array_neg))
y = np.concatenate((np.ones((len(features_array_pos),)),np.zeros((len(features_array_neg),))))
weights = np.concatenate((weight_array_pos, weight_array_neg)) / 1000

# split your data
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# initialize and train
# n_estimators=100 means 100 decision trees will work together
model = RandomForestClassifier(n_estimators=100, n_jobs=-1) 
model.fit(X_train, y_train, sample_weight=w_train)

# evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))