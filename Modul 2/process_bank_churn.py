import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from typing import Tuple, List, Optional, Dict, Any


TARGET_COL = "Exited"
COLUMNS_TO_EXCLUDE = ["CustomerId", "Surname", "id"]


def select_input_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select input columns excluding identifiers and target.

    Args:
        df: Raw dataframe.

    Returns:
        Tuple containing dataframe with input columns and list of column names.
    """
    exclude = set(COLUMNS_TO_EXCLUDE + [TARGET_COL])
    input_cols = [c for c in df.columns if c not in exclude]

    return df[input_cols].copy(), input_cols


def split_train_val(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and validation sets.

    Args:
        df: Input dataframe.
        target_col: Target column.
        test_size: Validation size.
        random_state: Random seed.

    Returns:
        Train and validation dataframes.
    """

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    return train_df, val_df


def create_inputs_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    input_cols: List[str],
    target_col: str,
):
    """
    Create inputs and targets for train and validation.

    Returns:
        train_inputs, train_targets, val_inputs, val_targets
    """

    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()

    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()

    return train_inputs, train_targets, val_inputs, val_targets


def impute_missing_values(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    numeric_cols: List[str],
) -> SimpleImputer:
    """
    Impute missing numeric values.

    Returns:
        Fitted SimpleImputer.
    """

    imputer = SimpleImputer(strategy="mean")

    train_inputs[numeric_cols] = imputer.fit_transform(
        train_inputs[numeric_cols]
    )

    val_inputs[numeric_cols] = imputer.transform(
        val_inputs[numeric_cols]
    )

    return imputer


def scale_numeric_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    numeric_cols: List[str],
) -> MinMaxScaler:
    """
    Scale numeric features.

    Returns:
        Fitted scaler.
    """

    scaler = MinMaxScaler()

    train_inputs[numeric_cols] = scaler.fit_transform(
        train_inputs[numeric_cols]
    )

    val_inputs[numeric_cols] = scaler.transform(
        val_inputs[numeric_cols]
    )

    return scaler


def encode_categorical_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    categorical_cols: List[str],
):
    """
    One-hot encode categorical features.

    Returns:
        encoder and encoded column names.
    """

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )

    encoder.fit(train_inputs[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    train_encoded = encoder.transform(train_inputs[categorical_cols])
    val_encoded = encoder.transform(val_inputs[categorical_cols])

    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoded_cols,
        index=train_inputs.index,
    )

    val_encoded_df = pd.DataFrame(
        val_encoded,
        columns=encoded_cols,
        index=val_inputs.index,
    )

    train_inputs.drop(columns=categorical_cols, inplace=True)
    val_inputs.drop(columns=categorical_cols, inplace=True)

    train_inputs = pd.concat([train_inputs, train_encoded_df], axis=1)
    val_inputs = pd.concat([val_inputs, val_encoded_df], axis=1)

    return train_inputs, val_inputs, encoder, list(encoded_cols)


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = False,
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline for bank churn data.

    Args:
        raw_df: Raw dataframe with all columns.
        scaler_numeric: Whether to scale numeric features.

    Returns:
        Dictionary containing:
            - train_X: Training features
            - train_y: Training targets
            - val_X: Validation features
            - val_y: Validation targets
            - input_cols: List of final column names
            - numeric_cols: List of numeric column names
            - categorical_cols: List of categorical column names
            - imputer: Fitted SimpleImputer
            - scaler: Fitted MinMaxScaler (or None)
            - encoder: Fitted OneHotEncoder
    """

    inputs_df, input_cols = select_input_columns(raw_df)

    full_df = inputs_df.copy()
    full_df[TARGET_COL] = raw_df[TARGET_COL]

    train_df, val_df = split_train_val(full_df, TARGET_COL)

    train_inputs, train_targets, val_inputs, val_targets = (
        create_inputs_targets(
            train_df,
            val_df,
            input_cols,
            TARGET_COL,
        )
    )

    numeric_cols = train_inputs.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = train_inputs.select_dtypes(
        include="object"
    ).columns.tolist()

    imputer = impute_missing_values(
        train_inputs,
        val_inputs,
        numeric_cols,
    )

    scaler = None
    if scaler_numeric:
        scaler = scale_numeric_features(
            train_inputs,
            val_inputs,
            numeric_cols,
        )

    train_inputs, val_inputs, encoder, encoded_cols = (
        encode_categorical_features(
            train_inputs,
            val_inputs,
            categorical_cols,
        )
    )

    final_cols = numeric_cols + encoded_cols

    X_train = train_inputs[final_cols]
    X_val = val_inputs[final_cols]

    return {
    "train_X": X_train,
    "train_y": train_targets,
    "val_X": X_val,
    "val_y": val_targets,
    "input_cols": final_cols,
    "numeric_cols": numeric_cols,       
    "categorical_cols": categorical_cols, 
    "imputer": imputer,                 
    "scaler": scaler,
    "encoder": encoder,
}


def preprocess_new_data(
    new_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    imputer: SimpleImputer,
    encoder: OneHotEncoder,
    scaler: Optional[MinMaxScaler] = None,
) -> pd.DataFrame:
    """
    Process new data using pre-fitted transformers.

    Args:
        new_df: New DataFrame (e.g., test.csv).
        numeric_cols: List of numeric columns (before encoding).
        categorical_cols: List of categorical columns.
        imputer: Fitted SimpleImputer for numeric features.
        encoder: Fitted OneHotEncoder for categorical features.
        scaler: Fitted MinMaxScaler (optional).

    Returns:
        Processed DataFrame with the same column order as during training.
    """
    X = new_df[numeric_cols + categorical_cols].copy()

    X[numeric_cols] = imputer.transform(X[numeric_cols])

    if scaler is not None:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    encoded = encoder.transform(X[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_cols,
        index=X.index,
    )

    X = X.drop(columns=categorical_cols)
    X = pd.concat([X, encoded_df], axis=1)

    return X[numeric_cols + list(encoded_cols)]
