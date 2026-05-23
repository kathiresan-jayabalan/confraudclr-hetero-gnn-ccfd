# ============================================================
# CREDIT CARD FRAUD DETECTION - COMMON EXPERIMENTAL SETUP
# ============================================================
# Dataset:
# Kaggle Credit Card Fraud Detection Dataset
#
# Common Experimental Configuration:
# - Train/Test Split: 70:30
# - Random State: 42
# - Feature Scaling: StandardScaler
# - Data Balancing: SMOTE-ENN
# - Evaluation Metrics:
#   Accuracy, Precision, Recall, F1-Score
#
# Models Included:
# 1. Decision Tree (DT)
# 2. Logistic Regression (LR)
# 3. Random Forest (RF)
# 4. CatBoost
# 5. Adaptive Federated Learning (Simplified AFL)
# 6. XGBoost
# 7. Naive Bayes (NB)
# 8. VAE-GAT-XGBoost (Simplified)
# 9. CNN-LSTM
# 10. RF-LSTM
# 11. FraudX-AI (XGBoost + RF Ensemble)
# 12. HMOA-GNN (Simplified GNN)
# 13. Multimodal Neural Network (MNN)
# 14. Neural Network (NN)
# 15. CBLOF
# 16. Adversarial Autoencoder (AAE)
# 17. LGBM
# ============================================================

# ==========================
# INSTALL REQUIRED LIBRARIES
# ==========================
# pip install pandas numpy scikit-learn imbalanced-learn
# pip install xgboost lightgbm catboost pyod
# pip install tensorflow torch torch-geometric

# ==========================
# IMPORT LIBRARIES
# ==========================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.combine import SMOTEENN

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Classical ML
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Boosting
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Outlier Detection
from pyod.models.cblof import CBLOF

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import concatenate

# ==========================
# LOAD DATASET
# ==========================
# Replace path with your dataset location
data = pd.read_csv("creditcard.csv")

# ==========================
# FEATURE / LABEL SPLIT
# ==========================
X = data.drop("Class", axis=1)
y = data["Class"]

# ==========================
# FEATURE SCALING
# ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# TRAIN TEST SPLIT (70:30)
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ==========================
# SMOTE-ENN BALANCING
# ==========================
smote_enn = SMOTEENN(random_state=42)

X_train_balanced, y_train_balanced = smote_enn.fit_resample(
    X_train,
    y_train
)

print("Balanced Training Shape:", X_train_balanced.shape)

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_model(name, y_true, y_pred):

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n==============================")
    print("Model:", name)
    print("==============================")
    print("Accuracy :", round(acc * 100, 4))
    print("Precision:", round(prec * 100, 4))
    print("Recall   :", round(rec * 100, 4))
    print("F1-Score :", round(f1 * 100, 4))

# ============================================================
# 1. DECISION TREE (DT)
# ============================================================
dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train_balanced, y_train_balanced)

dt_pred = dt.predict(X_test)

evaluate_model("Decision Tree", y_test, dt_pred)

# ============================================================
# 2. LOGISTIC REGRESSION (LR)
# ============================================================
lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_balanced, y_train_balanced)

lr_pred = lr.predict(X_test)

evaluate_model("Logistic Regression", y_test, lr_pred)

# ============================================================
# 3. RANDOM FOREST (RF)
# ============================================================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_balanced, y_train_balanced)

rf_pred = rf.predict(X_test)

evaluate_model("Random Forest", y_test, rf_pred)

# ============================================================
# 4. CATBOOST
# ============================================================
cat = CatBoostClassifier(
    iterations=100,
    verbose=0,
    random_seed=42
)

cat.fit(X_train_balanced, y_train_balanced)

cat_pred = cat.predict(X_test)

evaluate_model("CatBoost", y_test, cat_pred)

# ============================================================
# 5. ADAPTIVE FEDERATED LEARNING (SIMPLIFIED AFL)
# ============================================================
# Simulated using MLP

afl = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=100,
    random_state=42
)

afl.fit(X_train_balanced, y_train_balanced)

afl_pred = afl.predict(X_test)

evaluate_model("Adaptive Federated Learning", y_test, afl_pred)

# ============================================================
# 6. XGBOOST
# ============================================================
xgb = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

xgb.fit(X_train_balanced, y_train_balanced)

xgb_pred = xgb.predict(X_test)

evaluate_model("XGBoost", y_test, xgb_pred)

# ============================================================
# 7. NAIVE BAYES (NB)
# ============================================================
nb = GaussianNB()

nb.fit(X_train_balanced, y_train_balanced)

nb_pred = nb.predict(X_test)

evaluate_model("Naive Bayes", y_test, nb_pred)

# ============================================================
# 8. VAE-GAT-XGBOOST (SIMPLIFIED)
# ============================================================
# Simplified as Autoencoder + XGBoost

input_dim = X_train_balanced.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)

encoder = Model(input_layer, encoded)

autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

autoencoder.fit(
    X_train_balanced,
    X_train_balanced,
    epochs=10,
    batch_size=256,
    verbose=0
)

X_train_encoded = encoder.predict(X_train_balanced)
X_test_encoded = encoder.predict(X_test)

vae_xgb = XGBClassifier(
    eval_metric='logloss'
)

vae_xgb.fit(X_train_encoded, y_train_balanced)

vae_xgb_pred = vae_xgb.predict(X_test_encoded)

evaluate_model("VAE-GAT-XGBoost", y_test, vae_xgb_pred)

# ============================================================
# 9. CNN-LSTM
# ============================================================
X_train_dl = np.expand_dims(X_train_balanced, axis=2)
X_test_dl = np.expand_dims(X_test, axis=2)

cnn_lstm = Sequential()

cnn_lstm.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        input_shape=(X_train_dl.shape[1], 1)
    )
)

cnn_lstm.add(MaxPooling1D(pool_size=2))

cnn_lstm.add(LSTM(64))

cnn_lstm.add(Dense(1, activation='sigmoid'))

cnn_lstm.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

cnn_lstm.fit(
    X_train_dl,
    y_train_balanced,
    epochs=5,
    batch_size=256,
    verbose=0
)

cnn_lstm_pred = (
    cnn_lstm.predict(X_test_dl) > 0.5
).astype(int)

evaluate_model(
    "CNN-LSTM",
    y_test,
    cnn_lstm_pred
)

# ============================================================
# 10. RF-LSTM
# ============================================================
rf_features = rf.predict_proba(X_train_balanced)

rf_lstm_input = np.expand_dims(rf_features, axis=2)

rf_lstm = Sequential()

rf_lstm.add(
    LSTM(
        64,
        input_shape=(rf_lstm_input.shape[1], 1)
    )
)

rf_lstm.add(Dense(1, activation='sigmoid'))

rf_lstm.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

rf_lstm.fit(
    rf_lstm_input,
    y_train_balanced,
    epochs=5,
    batch_size=256,
    verbose=0
)

test_rf_features = rf.predict_proba(X_test)

test_rf_features = np.expand_dims(
    test_rf_features,
    axis=2
)

rf_lstm_pred = (
    rf_lstm.predict(test_rf_features) > 0.5
).astype(int)

evaluate_model(
    "RF-LSTM",
    y_test,
    rf_lstm_pred
)

# ============================================================
# 11. FRAUDX-AI (RF + XGBOOST ENSEMBLE)
# ============================================================
rf_prob = rf.predict_proba(X_test)[:, 1]
xgb_prob = xgb.predict_proba(X_test)[:, 1]

ensemble_prob = (rf_prob + xgb_prob) / 2

fraudx_pred = (ensemble_prob > 0.5).astype(int)

evaluate_model(
    "FraudX-AI",
    y_test,
    fraudx_pred
)

# ============================================================
# 12. HMOA-GNN (SIMPLIFIED)
# ============================================================
# Simplified using deep MLP

gnn = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    max_iter=100,
    random_state=42
)

gnn.fit(X_train_balanced, y_train_balanced)

gnn_pred = gnn.predict(X_test)

evaluate_model("HMOA-GNN", y_test, gnn_pred)

# ============================================================
# 13. MULTIMODAL NEURAL NETWORK (MNN)
# ============================================================
input1 = Input(shape=(15,))
input2 = Input(shape=(15,))

dense1 = Dense(64, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(input2)

merged = concatenate([dense1, dense2])

output = Dense(1, activation='sigmoid')(merged)

mnn = Model(
    inputs=[input1, input2],
    outputs=output
)

mnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

X1_train = X_train_balanced[:, :15]
X2_train = X_train_balanced[:, 15:30]

X1_test = X_test[:, :15]
X2_test = X_test[:, 15:30]

mnn.fit(
    [X1_train, X2_train],
    y_train_balanced,
    epochs=5,
    batch_size=256,
    verbose=0
)

mnn_pred = (
    mnn.predict([X1_test, X2_test]) > 0.5
).astype(int)

evaluate_model(
    "Multimodal Neural Network",
    y_test,
    mnn_pred
)

# ============================================================
# 14. NEURAL NETWORK (NN)
# ============================================================
nn = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=100,
    random_state=42
)

nn.fit(X_train_balanced, y_train_balanced)

nn_pred = nn.predict(X_test)

evaluate_model("Neural Network", y_test, nn_pred)

# ============================================================
# 15. CBLOF
# ============================================================
cblof = CBLOF(
    contamination=0.001
)

cblof.fit(X_train_balanced)

cblof_pred = cblof.predict(X_test)

evaluate_model(
    "CBLOF",
    y_test,
    cblof_pred
)

# ============================================================
# 16. ADVERSARIAL AUTOENCODER (AAE)
# ============================================================
aae_input = Input(shape=(input_dim,))

x = Dense(128, activation='relu')(aae_input)
x = Dense(64, activation='relu')(x)
encoded = Dense(32, activation='relu')(x)

x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)

decoded = Dense(input_dim, activation='linear')(x)

aae = Model(aae_input, decoded)

aae.compile(
    optimizer='adam',
    loss='mse'
)

aae.fit(
    X_train_balanced,
    X_train_balanced,
    epochs=10,
    batch_size=256,
    verbose=0
)

reconstructed = aae.predict(X_test)

reconstruction_error = np.mean(
    np.square(X_test - reconstructed),
    axis=1
)

threshold = np.percentile(
    reconstruction_error,
    95
)

aae_pred = (
    reconstruction_error > threshold
).astype(int)

evaluate_model(
    "Adversarial Autoencoder",
    y_test,
    aae_pred
)

# ============================================================
# 17. LIGHTGBM (LGBM)
# ============================================================
lgbm = LGBMClassifier(
    random_state=42
)

lgbm.fit(
    X_train_balanced,
    y_train_balanced
)

lgbm_pred = lgbm.predict(X_test)

evaluate_model(
    "LightGBM",
    y_test,
    lgbm_pred
)

# ============================================================
# END OF ALL MODELS
# ============================================================
