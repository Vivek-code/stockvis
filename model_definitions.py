from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout, Input,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam

def create_lstm_model(input_shape):
    """
    Creates an LSTM model architecture.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1) # Linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_gru_model(input_shape):
    """
    Creates a GRU model architecture.
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_cnn_model(input_shape):
    """
    Creates a 1D-CNN model architecture.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='causal'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_transformer_model(input_shape):
    """
    Creates a Transformer Encoder model for time-series regression.
    Uses MultiHeadAttention with residual connections and LayerNormalization.
    """
    inputs = Input(shape=input_shape)

    # Project input features to d_model dimensions
    x = Dense(64)(inputs)

    # Transformer Encoder Block 1
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    attn_output = Dropout(0.2)(attn_output)
    x = LayerNormalization()(x + attn_output)  # Residual connection

    ff_output = Dense(128, activation='relu')(x)
    ff_output = Dropout(0.2)(ff_output)
    ff_output = Dense(64)(ff_output)
    x = LayerNormalization()(x + ff_output)  # Residual connection

    # Transformer Encoder Block 2
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    attn_output = Dropout(0.2)(attn_output)
    x = LayerNormalization()(x + attn_output)

    ff_output = Dense(128, activation='relu')(x)
    ff_output = Dropout(0.2)(ff_output)
    ff_output = Dense(64)(ff_output)
    x = LayerNormalization()(x + ff_output)

    # Collapse the time dimension
    x = GlobalAveragePooling1D()(x)

    # Classification / Regression head
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
    return model
