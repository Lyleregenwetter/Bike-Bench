from typing import Callable, List
import numpy as np
import pandas as pd

# columns to one‐hot encode
ONE_HOT_ENCODED_BIKEBENCH_COLUMNS: List[str] = [
    'MATERIAL',
    'Head tube type',
    'Down tube type',
    'RIM_STYLE front',
    'RIM_STYLE rear',
    'Handlebar style',
    'Stem kind',
    'Fork type',
    'Seat tube type',
]

ALL_CATEGORIES = {
    'MATERIAL': [
        'ALUMINIUM',
        'STEEL',
        'TITANIUM'
    ],
    'Head tube type': [
        '0',
        '1',
        '2',
        '3'
    ],
    'Down tube type': [
        '0',
        '1',
        '2'
    ],
    'RIM_STYLE front': [
        'DISC',
        'SPOKED',
        'TRISPOKE'
    ],
    'RIM_STYLE rear': [
        'DISC',
        'SPOKED',
        'TRISPOKE'
    ],
    'Handlebar style': [
        '0',
        '1',
        '2'
    ],
    'Stem kind': [
        '0',
        '1',
        '2'
    ],
    'Fork type': [
        '0',
        '1',
        '2'
    ],
    'Seat tube type': [
        '0',
        '1',
        '2'
    ]
}

def normalize_category_value(value):
    """
    Normalize category values:
    - Convert float representations of integers (e.g., 1.0) to int strings ('1').
    - Convert NaN to 'nan'.
    - Strip whitespace.
    """
    if pd.isna(value):
        return 'nan'
    elif isinstance(value, (float, int)):
        # Check if it's an integer-looking float like 1.0
        if float(value).is_integer():
            return str(int(value))
        else:
            return str(value).strip()
    else:
        return str(value).strip()

# columns that are already boolean and should stay in the DF (converted to float on encode)
BOOLEAN_COLUMNS: List[str] = [
    'bottle SEATTUBE0 show',
    'bottle DOWNTUBE0 show',
    'BELTorCHAIN',
    'SEATSTAYbrdgCheck',
    'CHAINSTAYbrdgCheck',
]

FAKE_BOOLEAN_COLUMNS: List[str] = ['BELTorCHAIN']

PREFIX_SEP = " OHCLASS: "


def encode_to_continuous(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    for col in ONE_HOT_ENCODED_BIKEBENCH_COLUMNS:
        # Normalize the column values
        out[col] = out[col].apply(normalize_category_value)

        all_categories = set(ALL_CATEGORIES.get(col, []))
        present_categories = set(out[col].unique())

        unknown_categories = present_categories - all_categories
        if unknown_categories:
            print(f"⚠️ Warning: Column '{col}' has unknown values: {unknown_categories}")

        # Proceed with known values only
        dummies = pd.get_dummies(out[col], prefix=col, prefix_sep=PREFIX_SEP)

        # Ensure all known categories are represented
        for category in all_categories:
            category_col = f"{col}{PREFIX_SEP}{category}"
            if category_col not in dummies.columns:
                dummies[category_col] = 0

        # Reorder columns
        ordered_cols = [f"{col}{PREFIX_SEP}{cat}" for cat in sorted(all_categories)]
        dummies = dummies.reindex(columns=ordered_cols, fill_value=0)

        # Replace original column with dummies
        out = pd.concat([out.drop(columns=[col]), dummies], axis=1)

    # Convert booleans to float
    for col in BOOLEAN_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(float)
            
    #replace any non-floats with 0.0
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0)
            print(f"⚠️ Warning: Column '{col}' contained non-numeric values which were converted to 0.0")

    return out.astype(np.float32)



def decode_to_mixed(encoded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reverse the one‐hot encoding:
    - For each original categorical column, find all "<col> OHCLASS: *" dummies,
      take argmax (the position of the highest value), strip off the prefix, and restore
      the category string.
    - Round the float boolean columns back to 0/1 and cast to bool.
    """
    out = encoded_df.copy(deep=True)

    # 1) decode each categorical variable
    for col in ONE_HOT_ENCODED_BIKEBENCH_COLUMNS:
        pref = f"{col}{PREFIX_SEP}"
        dummy_cols = [c for c in out.columns if c.startswith(pref)]
        if not dummy_cols:
            continue

        # idxmax on the raw floats picks the column with the highest value
        restored = (
            out[dummy_cols]
            .idxmax(axis=1)
            .str.replace(pref, "", n=1, regex=False)
        )

        out[col] = restored
        out.drop(columns=dummy_cols, inplace=True)

    # 2) round boolean floats back to bool
    for col in BOOLEAN_COLUMNS:
        if col in out.columns and col not in FAKE_BOOLEAN_COLUMNS:
            out[col] = out[col].round().astype(int).astype(bool)

    return out


