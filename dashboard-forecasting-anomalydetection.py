import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Convert 'DateTime' column to datetime
def load_data(file):
    df = pd.read_excel(file)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df

def train_and_visualize(df, feature):
    cols = [feature]

    df_for_training = df[cols].astype(float)

    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)

    n_future = 1
    n_past = 14

    trainX = []
    trainY = []

    for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(50, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    model.fit(trainX, trainY, epochs=12, batch_size=16, validation_split=0.1, verbose=1)

    n_days_for_prediction = 50
    predict_period_dates = pd.date_range(df['DateTime'].iloc[-n_past], periods=n_days_for_prediction, freq='D')
    prediction = model.predict(trainX[-n_days_for_prediction:])
    y_pred_future = scaler.inverse_transform(prediction)[:, 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['DateTime'], df[feature], label='Actual')
    ax.plot(predict_period_dates, y_pred_future, label='Forecast')
    ax.set_xlabel('Time')
    ax.set_ylabel(feature)
    ax.legend()
    ax.set_title(f'{feature} - Actual vs Forecast (Zoomed)')
    ax.set_xlim(df['DateTime'].iloc[-n_past], predict_period_dates[-1])

    return fig

def anomaly_detection(df, features):
    clf = IsolationForest(contamination=0.00211, random_state=42)
    anomaly_plots = []

    for feature in features:
        X = df[[feature]]
        clf.fit(X)
        df[f'anomaly_{feature}'] = clf.predict(X)

    fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(15, 5 * len(features)))

    if len(features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        ax.plot(df.index, df[feature], label='Data Asli')
        anomalies = df[df[f'anomaly_{feature}'] == -1]
        ax.scatter(anomalies.index, anomalies[feature], color='red', label='Anomali')
        ax.fill_between(df.index, df[feature].min(), df[feature].max(), where=df[f'anomaly_{feature}'] == -1, color='red', alpha=0.3)
        ax.set_title(f'Deteksi Anomali pada Fitur {feature} menggunakan Isolation Forest')
        ax.set_xlabel('Index Data')
        ax.set_ylabel(f'Nilai {feature}')
        ax.legend()

    plt.tight_layout()
    return fig

# Streamlit app
st.title('ML Model for Forecasting and Anomaly Detection')
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write(df.head())

    features = st.multiselect('Select features to forecast', df.columns[1:])
    if st.button('Train and Visualize'):
        for feature in features:
            fig = train_and_visualize(df, feature)
            st.pyplot(fig)

        fig = anomaly_detection(df, features)
        st.pyplot(fig)
