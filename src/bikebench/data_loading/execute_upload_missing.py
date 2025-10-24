# upload_missing.py
from pathlib import Path
from bikebench.resource_utils import datasets_path
from bikebench.data_loading import dataverse_utils

import dataverse_utils as dv
import os

DOI = "10.7910/DVN/BSJSM6"

DV_API = os.environ.get("DATAVERSE_API_URL", "https://dataverse.harvard.edu/api")
os.environ["DATAVERSE_API_TOKEN"] = "3c34d30b-da7e-46ba-af84-67c529f0679a"

# dataverse_utils.upload_directory("10.7910/DVN/BSJSM6",
#                  datasets_path("Synthetic_Extended_Data/CTGAN/embeddings"),
#                  dv_prefix="Synthetic_Extended_Data/CTGAN/embeddings",
#                  replace_existing=False)

dataverse_utils.upload_directory("10.7910/DVN/BSJSM6",
                 datasets_path("Generative_Modeling_Datasets"),
                 dv_prefix="Generative_Modeling_Datasets",
                 replace_existing=False)