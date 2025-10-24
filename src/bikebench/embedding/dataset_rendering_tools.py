import logging
import os.path
import time
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from sdv.single_table import CTGANSynthesizer

from typing import Dict, Tuple, List
import sys
import pandas as pd
import cairosvg
import torch.nn.functional as F

import os
from PIL import Image
import numpy as np
import torch
import shutil
import uuid



from bikebench.rendering.rendering import RenderingEngine, FILE_BUILDER
from bikebench.data_loading import data_loading
from bikebench.resource_utils import resource_path, STANDARD_BIKE_RESOURCE, models_and_scalers_path
from bikebench.embedding.clip_embedding_calculator import ClipEmbeddingCalculator
from bikebench.transformation.one_hot_encoding import encode_to_continuous
from bikebench.validation.base_validation_function import construct_tensor_validator
from bikebench.validation.bike_bench_validation_functions import bike_bench_validation_functions



def read_standard_xml():
    with open(STANDARD_BIKE_RESOURCE, "r") as file:
        return file.read()

standard_bike_xml = read_standard_xml()

def get_bike_bench_records_with_id(num) -> Dict[str, dict]:
    """
    Return records in a dictionary of the form {
    (record_id: str) : (record: dict)
    }
    """
    data_train = data_loading.load_bike_bench_train()
    data_test = data_loading.load_bike_bench_test()
    data = pd.concat([data_train, data_test], axis=0)
    if num is None:
        num = len(data)
    else:
        data = data.iloc[:num, :]

    return {str(record_id): record for record_id, record in zip(data.index.tolist(), data.to_dict(orient="records"))}

from bikebench.validation.base_validation_function import construct_tensor_validator


def sample_CTGAN(n=4096) -> pd.DataFrame:
    save_path = models_and_scalers_path("CTGAN.pkl")
    synthesizer = CTGANSynthesizer.load(filepath=save_path)
    synthetic_collapsed = synthesizer.sample(num_rows=n)
    return synthetic_collapsed

def sample_uniform(n: int, p: float = 1.0, seed: int | None = None) -> pd.DataFrame:
    df = data_loading.load_bike_bench_mixed_modality_train()
    rng = np.random.default_rng(seed)
    out = {}

    for c in df.columns:
        s = df[c]

        # Booleans: uniform across classes if both present
        if s.dtype == bool:
            vals = s.dropna().unique()
            if len(vals) == 0:
                out[c] = pd.Series([pd.NA] * n, dtype="boolean")
            elif len(vals) == 1:
                out[c] = pd.Series([bool(vals[0])] * n, dtype="boolean")
            else:
                out[c] = pd.Series(rng.integers(0, 2, size=n).astype(bool), dtype="boolean")
            continue

        # Floats: uniform within percentile band
        if pd.api.types.is_float_dtype(s):
            clean = pd.to_numeric(s, errors="coerce").dropna()
            if len(clean) == 0:
                out[c] = pd.Series([np.nan] * n, dtype=float)
            else:
                lo, hi = np.percentile(clean, [p, 100 - p])
                draw = np.full(n, lo) if not np.isfinite(hi) or hi <= lo else rng.uniform(lo, hi, size=n)
                out[c] = pd.Series(draw, dtype=float)
            continue

        # Ints / objects / categoricals: sample uniformly over observed values
        vals = pd.unique(s.dropna())
        if len(vals) == 0:
            out[c] = pd.Series([np.nan] * n, dtype=(s.dtype if pd.api.types.is_integer_dtype(s) else object))
        else:
            draw = rng.choice(vals, size=n, replace=True)
            if pd.api.types.is_integer_dtype(s):
                out[c] = pd.Series(draw).astype(s.dtype)
            elif pd.api.types.is_categorical_dtype(s):
                out[c] = pd.Categorical(draw, categories=s.cat.categories)
            else:
                out[c] = pd.Series(draw, dtype=object)

    return pd.DataFrame(out, columns=df.columns)

def sample_n(n, method) -> pd.DataFrame:
    if method == "CTGAN":
        sampler_fn = sample_CTGAN
    elif method == "uniform":
        sampler_fn = sample_uniform
    else:
        raise ValueError("method must be 'CTGAN' or 'uniform'")
    sample_datapoint = sampler_fn(1)
    sample_datapoint_oh = encode_to_continuous(sample_datapoint)
    COLUMN_NAMES = list(sample_datapoint_oh.columns)
    tensor_validator, validation_names = construct_tensor_validator(bike_bench_validation_functions, COLUMN_NAMES)

    all_valid_samples = None
    while True:
        synthetic_collapsed = sampler_fn(10000)
        samples_oh = encode_to_continuous(synthetic_collapsed)
        samples_oh_tens = torch.tensor(samples_oh.values, dtype=torch.float32)

        validity = tensor_validator(samples_oh_tens)

        valid = torch.all(validity<=0, dim=1)
        valid_subset = samples_oh_tens[valid, :]
        if all_valid_samples is None:
            all_valid_samples = valid_subset
        else:
            all_valid_samples = torch.cat((all_valid_samples, valid_subset), dim=0)
        if all_valid_samples.shape[0] >= n:
            break
    all_valid_samples = all_valid_samples[:n, :]
    samples_df = pd.DataFrame(all_valid_samples.numpy(), columns=COLUMN_NAMES)
    return samples_df

def sample_save_n_records(save_path, n=4096, method = "CTGAN") -> Dict[str, dict]:
    data = sample_n(n, method)
    #make the data indices random keys using uuid
    random_keys = [str(uuid.uuid4()) for _ in range(len(data))]
    data.index = random_keys
    #save csv to save_path
    data.to_csv(save_path)
    return {str(record_id): record for record_id, record in zip(data.index.tolist(), data.to_dict(orient="records"))}

def bike_to_xml(save_path: str, record_id: str, record: dict):
    try:
        file_path = os.path.join(save_path, f"{record_id}.bcad")
        with open(file_path, "w") as file:
            xml_data = FILE_BUILDER.build_cad_from_clip(record, standard_bike_xml)
            file.write(xml_data)
    except Exception as e:
        print(f"XML Conversion Failed with exception {e}")


def bikes_to_xmls(records_with_id: Dict[str, dict],
                   process_pool_workers: int,
                   save_dir: str
                   ):
    executor = ProcessPoolExecutor(max_workers=process_pool_workers)
    os.makedirs(save_dir, exist_ok=True)
    for record_id, record in records_with_id.items():
        executor.submit(bike_to_xml, save_dir, record_id, record)
    executor.shutdown()  # waits for all submitted tasks to finish

def xmls_to_svgs(
        thread_pool_workers: int,
        records_with_id: Dict[str, dict],
        xml_dir: str,
        svg_dir: str,
        rendering_engine: RenderingEngine
):
    executor = ThreadPoolExecutor(max_workers=thread_pool_workers)

    os.makedirs(svg_dir, exist_ok=True)
    def xml_to_svg(xml: str):
        try:
            xml_path = os.path.join(xml_dir, f"{xml}.bcad")
            with open(xml_path, "r") as xml_file:
                # print("Sending request to server...")
                read_file = xml_file.read()
                # print("Read file...")
                rendering_result = rendering_engine.render_xml(read_file)
                # print("Rendering result received from server...")
                image_path = os.path.join(svg_dir, f"{xml}.svg")
                with open(image_path, "wb") as image_file:
                    image_file.write(rendering_result.image_bytes)
                return True
        except Exception as e:
            print(f"Rendering failed: {e}")
            return False, e

    for record_id, _ in records_with_id.items():
        executor.submit(xml_to_svg, record_id)

    executor.shutdown()


def svg_to_png(record_id: str, svg_dir: str, png_dir: str):
        try:
            svg_file = os.path.join(svg_dir, f"{record_id}.svg")
            png_file = os.path.join(png_dir, f"{record_id}.png")
            cairosvg.svg2png(url=svg_file, write_to=png_file)
        except Exception as e:
            print(f"Failed to convert {record_id} with exception {e}")

def svgs_to_pngs(process_pool_workers: int,
               records_with_id: Dict[str, dict],
                svg_dir: str,
               png_dir: str):
    executor = ProcessPoolExecutor(max_workers=process_pool_workers)
    os.makedirs(png_dir, exist_ok=True)
    for record_id, _ in records_with_id.items():
        executor.submit(svg_to_png, record_id, svg_dir, png_dir)
    executor.shutdown()  # waits for all submitted tasks to finish
    
#load all pngs and stack up
def load_pngs(png_dir: str,
              records_with_id: Dict[str, dict]) -> Tuple[torch.Tensor, List[str]]:
    """
    Scans png_dir for files named <record_id>.png (in the order of records_with_id.keys()),
    loads each as an RGB image, converts to a float tensor in [0,1],
    pads on TOP and RIGHT to the max H/W across the set,
    and stacks them into a tensor of shape (N, 3, H_max, W_max).
    Returns (tensor, names).
    """
    PAD_VALUE = 1.0  # use 0.0 for black padding

    imgs = []
    names = []
    heights = []
    widths = []

    for record_id in records_with_id:
        path = os.path.join(png_dir, f"{record_id}.png")
        if not os.path.exists(path):
            print(f"Warning: {path} not found; skipping.")
            continue

        with Image.open(path) as img:
            img = img.convert("RGB")            # ensure 3 channels
            arr = np.array(img)                 # H x W x 3, uint8
            t = torch.from_numpy(arr).permute(2, 0, 1).float()  # 3 x H x W
            imgs.append(t)
            names.append(record_id)
            heights.append(t.shape[1])
            widths.append(t.shape[2])

    if not imgs:
        return torch.empty((0, 3, 0, 0)), []

    H_max = max(heights)
    W_max = max(widths)

    padded = []
    for t in imgs:
        _, H, W = t.shape
        pad_top = H_max - H
        pad_right = W_max - W
        # Pad order for 3D (C,H,W): (left, right, top, bottom)
        t_pad = F.pad(t, (0, pad_right, pad_top, 0), mode="constant", value=PAD_VALUE)
        padded.append(t_pad)

    return torch.stack(padded, dim=0), names


def embed_pngs(
    png_dir: str,
    records_with_id: Dict[str, dict],
    batch_size: int = 32,
    emb_file: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_embedder = ClipEmbeddingCalculator(batch_size=batch_size, device=device)

    all_embs = []
    all_names = []
    ids = list(records_with_id)

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        subset = {rid: records_with_id[rid] for rid in batch_ids}
        imgs, names = load_pngs(png_dir, subset)
        if not names:
            continue

        with torch.no_grad():
            emb = clip_embedder.embed_images(imgs).cpu()
        all_embs.append(emb)
        all_names.extend(names)

    embs = torch.cat(all_embs, dim=0).numpy()
    df = pd.DataFrame(embs, index=all_names)
    df.to_csv(emb_file or "embeddings.csv")


def process_rendering_stack(records, xml_dir: str, svg_dir: str, png_dir: str, emb_file: str, rendering_engine: RenderingEngine, process_pool_workers: int, thread_pool_workers: int):
    bikes_to_xmls(records, process_pool_workers, xml_dir)
    print("XMLs created")

    xmls_to_svgs(
        thread_pool_workers=thread_pool_workers,
        records_with_id=records,
        xml_dir=xml_dir,
        svg_dir=svg_dir,
        rendering_engine=rendering_engine,
    )
    print("SVGs created")
    svgs_to_pngs(process_pool_workers=process_pool_workers, records_with_id=records, svg_dir=svg_dir, png_dir=png_dir)
    print("PNGs created")
    embed_pngs(png_dir=png_dir, records_with_id=records, batch_size=32, emb_file=emb_file)
    print("Embeddings created")
    shutil.make_archive(xml_dir, 'gztar', xml_dir)
    shutil.make_archive(svg_dir, 'gztar', svg_dir)
    shutil.make_archive(png_dir, 'gztar', png_dir)
    print("Zipped all directories")
    shutil.rmtree(xml_dir)
    shutil.rmtree(svg_dir)
    shutil.rmtree(png_dir)
    print("Removed all directories")