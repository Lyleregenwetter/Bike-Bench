import pandas as pd
import torch
from typing import List


FRAMED_ORDERED_COLUMNS = ['Material=Steel', 'Material=Aluminum', 'Material=Titanium',
       'SSB_Include', 'CSB_Include', 'CS Length', 'BB Drop', 'Stack', 'SS E',
       'ST Angle', 'BB OD', 'TT OD', 'HT OD', 'DT OD', 'CS OD', 'SS OD',
       'ST OD', 'CS F', 'HT LX', 'ST UX', 'HT UX', 'HT Angle', 'HT Length',
       'ST Length', 'BB Length', 'Dropout Offset', 'SSB OD', 'CSB OD',
       'SSB Offset', 'CSB Offset', 'SS Z', 'SS Thickness', 'CS Thickness',
       'TT Thickness', 'BB Thickness', 'HT Thickness', 'ST Thickness',
       'DT Thickness', 'DT Length']

BIKEBENCH_TO_FRAMED_UNITS = {
    'ssd': 'SS OD',
    'Head tube length textfield': 'HT Length',
    'csd': 'CS OD',
    'Seat tube extension2': 'ST UX',
    'Head tube lower extension2': 'HT LX',
    'Head tube upper extension2': 'HT UX',
    'Seat tube length': 'ST Length',
    'BB textfield': 'BB Drop',
    'CHAINSTAYbrdgshift': 'CSB Offset',
    'Seat stay junction0': 'SS E',
    'Dropout spacing': 'Dropout Offset',
    'SSTopZOFFSET': 'SS Z',
    'Stack': 'Stack',
    'CS textfield': 'CS Length',
    'DT Length': 'DT Length',
    'htd': 'HT OD',
    'SEATSTAYbrdgshift': 'SSB Offset',
    'BB diameter': 'BB OD',
    'dtd': 'DT OD',
    'ttd': 'TT OD',
    'BB length': 'BB Length',
    'std': 'ST OD',
    'Wall thickness Seat stay': 'SS Thickness',
    'Wall thickness Chain stay': 'CS Thickness',
    'Wall thickness Top tube': 'TT Thickness',
    'Wall thickness Bottom Bracket': 'BB Thickness',
    'Wall thickness Head tube': 'HT Thickness',
    'Wall thickness Seat tube': 'ST Thickness',
    'Wall thickness Down tube': 'DT Thickness',
    'CHAINSTAYbrdgdia1': 'CSB OD',
    'SEATSTAYbrdgdia1': 'SSB OD',
    'Chain stay position on BB': 'CS F',
}

BIKEBENCH_TO_FRAMED_UNITS_OH = BIKEBENCH_TO_FRAMED_UNITS.copy()
BIKEBENCH_TO_FRAMED_UNITS_OH['MATERIAL OHCLASS: ALUMINIUM'] = 'Material=Aluminum'
BIKEBENCH_TO_FRAMED_UNITS_OH['MATERIAL OHCLASS: STEEL'] = 'Material=Steel'
BIKEBENCH_TO_FRAMED_UNITS_OH['MATERIAL OHCLASS: TITANIUM'] = 'Material=Titanium'


BIKEBENCH_TO_FRAMED_IDENTICAL = {
    'Head angle': 'HT Angle',
    'SEATSTAYbrdgCheck': 'SSB_Include',
    'Seat angle': 'ST Angle',
    'CHAINSTAYbrdgCheck': 'CSB_Include'
}


MATERIALS = {"MATERIAL OHCLASS: ALUMINIUM": "Aluminum",
             "MATERIAL OHCLASS: STEEL": "Steel",
             "MATERIAL OHCLASS: TITANIUM": "Titanium",
            }
def clip_to_framed(X_clip):
    X_framed = pd.DataFrame()
    #For every column in FRAMED_TO_BIKEBENCH_UNITS, add from X_clip to X_framed but divide by 1000 (mm to m)
    for column in BIKEBENCH_TO_FRAMED_UNITS.keys():
        if column in X_clip.columns:
            X_framed[BIKEBENCH_TO_FRAMED_UNITS[column]] = X_clip[column] / 1000.0

    #For every column in FRAMED_TO_BIKEBENCH_IDENTICAL, add from X_clip to X_framed but do not change units
    for column in BIKEBENCH_TO_FRAMED_IDENTICAL.keys():
        if column in X_clip.columns:
            X_framed[BIKEBENCH_TO_FRAMED_IDENTICAL[column]] = X_clip[column]

    #Conver one-hot encoded columns of the form MATERIAL OHCLASS: ALUMINIUM to a column Material with the value Aluminum
    material_columns = [col for col in X_clip.columns if col.startswith("MATERIAL OHCLASS:")]
    X_framed["Material"] = X_clip[material_columns].idxmax(axis=1).map(MATERIALS)

    return X_framed

def clip_to_framed_tensor_builder(clip_columns: List[str], framed_order: List[str]) -> callable:
    """
    clip_columns:  list of the CLIP‑DataFrame columns, in order,
                   so that X_clip tensor has shape [N, D] matching these.
    framed_order:  the *exact* list of framed‑column names you want out,
                   e.g. ["HT Length","ST Length",…,"Material"]
    Returns:
      fn(X_clip: Tensor[N,D]) -> Tensor[N, len(framed_order)]
      out_names == framed_order
    """

    # 1) find & name the unit‑converted cols
    units_idx, units_names = [], []
    ident_idx, ident_names = [], []
    for i, col in enumerate(clip_columns):
        if col in BIKEBENCH_TO_FRAMED_UNITS_OH:
            tgt = BIKEBENCH_TO_FRAMED_UNITS_OH[col]
            # treat Material=… as a one‑hot copy, not mm→m
            if tgt.startswith("Material="):
                ident_idx.append(i)
                ident_names.append(tgt)
            else:
                units_idx.append(i)
                units_names.append(tgt)

        elif col in BIKEBENCH_TO_FRAMED_IDENTICAL:
            ident_idx.append(i)
            ident_names.append(BIKEBENCH_TO_FRAMED_IDENTICAL[col])

    # base feature list in the order we'll build them
    base_names = units_names + ident_names

    # Step 2: figure out how to reorder base_names → framed_order
    reorder_idx = []
    for name in framed_order:
        if name not in base_names:
            raise ValueError(f"requested framed column {name!r} not found (got {base_names})")
        reorder_idx.append(base_names.index(name))

    def clip_to_framed_reordered(
        X_clip: torch.Tensor  # [N, D]
    ) -> torch.Tensor:      # [N, len(framed_order)]
        # mm→m conversion block
        block_units = X_clip[:, units_idx] / 1000.0       # [N, U]
        # direct‑copy block (includes Material=… one‑hots and IDENTICAL)
        block_ident = X_clip[:, ident_idx]                # [N, I]
        # concat in base order
        X_base = torch.cat([block_units, block_ident], dim=1)  # [N, U+I]
        # reorder to your target layout
        return X_base[:, reorder_idx]

    return clip_to_framed_reordered

    