import os, json, mimetypes, requests
from pathlib import Path, PurePosixPath
from typing import Dict, Tuple, List, Set, Optional
import time
from tqdm import tqdm

DV_API = os.environ.get("DATAVERSE_API_URL", "https://dataverse.harvard.edu/api")
DATA_API = os.environ.get("DATAVERSE_DATA_URL", "https://dataverse.harvard.edu/api/access/datafile")

# ---------------- basics ----------------

def _headers() -> Dict[str, str]:
    tok = os.environ.get("DATAVERSE_API_TOKEN")
    if not tok:
        raise RuntimeError("Set DATAVERSE_API_TOKEN")
    return {"X-Dataverse-key": tok}

def _with_pid(doi: str) -> str:
    return doi if doi.startswith(("doi:", "hdl:")) else f"doi:{doi}"

def _list_remote_files(doi: str) -> Dict[str, Dict]:
    """
    Return mapping: 'dir/label' -> {'id': int, 'size': int}
    Uses latestVersion of the dataset.
    """
    url = f"{DV_API}/datasets/:persistentId"
    r = requests.get(url, params={"persistentId": _with_pid(doi)}, headers=_headers(), timeout=60)
    # If you don't have perms for drafts, retry anonymously for published view:
    if r.status_code == 401:
        r = requests.get(url, params={"persistentId": _with_pid(doi)}, timeout=60)
    r.raise_for_status()
    out = {}
    for f in r.json()["data"]["latestVersion"].get("files", []):
        df = f["dataFile"]
        dirlabel = f.get("directoryLabel") or ""
        label = f.get("label") or ""
        key = f"{dirlabel}/{label}" if dirlabel else label
        out[key] = {"id": df["id"], "size": int(df.get("filesize", 0))}
    return out

def _walk_local_files(local_root: Path) -> List[Tuple[str, Path]]:
    """
    Return list of (posix_key, absolute_path) for all files under local_root.
    Keys are relative posix paths like 'sub/dir/file.ext'.
    """
    files: List[Tuple[str, Path]] = []
    for p in sorted(local_root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(local_root)
            files.append((PurePosixPath(rel).as_posix(), p))
    return files

# ---------------- pretty printing ----------------

def _tree_from_keys(keys: List[str]) -> dict:
    root = {}
    for k in keys:
        parts = [x for x in k.split("/") if x]
        cur = root
        for i, part in enumerate(parts):
            leaf = (i == len(parts) - 1)
            cur = cur.setdefault(part, {} if not leaf else None)  # None marks a file
    return root

def _print_tree(node: dict, prefix: str = ""):
    items = list(node.items())
    for i, (name, child) in enumerate(items):
        last = (i == len(items) - 1)
        branch = "└── " if last else "├── "
        print(prefix + branch + name)
        if isinstance(child, dict):
            _print_tree(child, prefix + ("    " if last else "│   "))

def print_remote_tree(doi: str) -> None:
    idx = _list_remote_files(doi)
    tree = _tree_from_keys(sorted(idx.keys()))
    print(f"[remote tree for {doi}]")
    _print_tree(tree)

def print_local_tree(local_root: str | Path) -> None:
    root = Path(local_root)
    keys = [k for k, _ in _walk_local_files(root)]
    tree = _tree_from_keys(keys)
    print(f"[local tree under {root}]")
    _print_tree(tree)

# ---------------- diff ----------------

def diff_local_vs_remote(local_root: str | Path, doi: str, dv_prefix: str = "") -> Dict[str, Set[str]]:
    """
    Compare local tree (under local_root) to remote tree (under optional dv_prefix).
    Returns dict with 'local_only' and 'remote_only' sets of keys (posix).
    """
    remote = _list_remote_files(doi)
    # limit remote to requested prefix (if any)
    if dv_prefix:
        dv_prefix = dv_prefix.rstrip("/") + "/"
        remote_keys = {k[len(dv_prefix):] for k in remote if k.startswith(dv_prefix)}
    else:
        remote_keys = set(remote.keys())

    local_keys = {k for k, _ in _walk_local_files(Path(local_root))}
    return {
        "local_only": local_keys - remote_keys,
        "remote_only": remote_keys - local_keys,
        # note: no "changed" category (checksums omitted by design)
    }

# ---------------- delete ----------------

def delete_remote_prefix(doi: str, prefix: str, dry_run: bool = False, sleep_between: float = 0.0) -> list[str]:
    """
    Delete all remote files whose key is exactly 'prefix' or starts with 'prefix/'.
    Works against the dataset's DRAFT (publish afterwards if you want the change live).
    Returns the list of deleted keys.
    """
    # Normalize matching
    pre = prefix.rstrip("/")
    idx = _list_remote_files(doi)
    targets = [(k, v["id"]) for k, v in idx.items() if (k == pre or k.startswith(pre + "/"))]

    if not targets:
        print(f"[delete] nothing to delete under '{pre}'")
        return []

    print(f"[delete] {len(targets)} files under '{pre}'")
    for key, fid in targets:
        print(f"DELETE {key} (id={fid})")
        if dry_run:
            continue
        r = requests.delete(f"{DV_API}/files/{fid}", headers=_headers(), timeout=60)
        r.raise_for_status()
        if sleep_between > 0:
            time.sleep(sleep_between)

    if dry_run:
        print("[dry-run] no deletions performed")
    return [k for k, _ in targets]


# ---------------- upload / replace ----------------

def _upload_new(doi: str, path: Path, directory_label: str) -> int:
    url = f"{DV_API}/datasets/:persistentId/add"
    params = {"persistentId": _with_pid(doi)}
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    data = {"jsonData": json.dumps({"directoryLabel": directory_label, "restrict": False})}
    with path.open("rb") as fh:
        files = {"file": (path.name, fh, mime)}
        r = requests.post(url, params=params, data=data, files=files, headers=_headers(), timeout=600)
    r.raise_for_status()
    return r.json()["data"]["files"][0]["dataFile"]["id"]

def _replace_file(file_id: int, path: Path, *, directory_label: Optional[str] = None, label: Optional[str] = None, force: bool = True, restrict: bool = False) -> int:
    url = f"{DV_API}/files/{file_id}/replace"
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

    meta = {"forceReplace": bool(force), "restrict": bool(restrict)}
    if directory_label is not None:
        meta["directoryLabel"] = directory_label  # <-- keep file in the same folder
    if label is not None:
        meta["label"] = label                      # optional; controls display name

    data = {"jsonData": json.dumps(meta)}

    with path.open("rb") as fh:
        files = {"file": (path.name, fh, mime)}
        r = requests.post(url, data=data, files=files, headers=_headers(), timeout=600)
    r.raise_for_status()
    return r.json()["data"]["files"][0]["dataFile"]["id"]

def upload_directory(
    doi: str,
    local_dir: str | Path,
    dv_prefix: str = "",
    replace_existing: bool = True,
) -> None:
    """
    Recursively upload local_dir into the dataset under dv_prefix.
    - If a remote file with the same directory/filename exists and replace_existing=True → REPLACE
      else ADD as new.
    """
    local_dir = Path(local_dir).resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(local_dir)

    remote = _list_remote_files(doi)

    for rel_key, abs_path in _walk_local_files(local_dir):
        # Compute Dataverse path pieces:
        #   remote key = f"{dv_prefix}/{rel_key}" (posix)
        full_key = f"{dv_prefix.rstrip('/')}/{rel_key}" if dv_prefix else rel_key
        directory = str(PurePosixPath(full_key).parent)
        directory = "" if directory == "." else directory

        # Decide ADD vs REPLACE
        if full_key in remote and replace_existing:
            fid = remote[full_key]["id"]
            print(f"REPLACE {full_key}  id={fid}")
            # compute the target folder and filename for metadata:
            directory = str(PurePosixPath(full_key).parent)
            directory = "" if directory == "." else directory
            filename = PurePosixPath(full_key).name
            _replace_file(fid, abs_path, directory_label=directory, label=filename)
        elif full_key in remote and not replace_existing:
            print(f"SKIP    {full_key} (exists; replace_existing=False)")
        else:
            print(f"ADD     {full_key}")
            _upload_new(doi, abs_path, directory)

def _access_params():
    tok = os.environ.get("DATAVERSE_API_TOKEN")
    return {"key": tok} if tok else {}

def download_by_key(doi: str, key: str, dest_path: str | Path) -> Path:
    """
    Resolve a dataset file by folder+name (key) and download it by id.
    Works for draft/restricted files when DATAVERSE_API_TOKEN is set.
    """
    idx = _list_remote_files(doi)
    if key not in idx:
        # try once more in case you just uploaded and the cache is stale
        idx = _list_remote_files(doi, refresh=True)
        if key not in idx:
            raise FileNotFoundError(f"Remote key not found: {key}")
    fid = idx[key]["id"]

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(
        f"{DATA_API}/{fid}",
        params=_access_params(),          # <-- pass token as query param
        stream=True,
        timeout=180,
        allow_redirects=True,
    ) as r:
        r.raise_for_status()
        tmp = dest_path.with_suffix(dest_path.suffix + ".part")
        with open(tmp, "wb") as out:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    out.write(chunk)
        tmp.rename(dest_path)
    return dest_path

def download_folder(doi: str, dv_prefix: str, dest_root: str | Path | None = None, overwrite: bool = False):
    """
    Download every Dataverse file whose key starts with dv_prefix + '/',
    preserving folder structure locally under dest_root (default: datasets_path).
    """

    dv_prefix = dv_prefix.rstrip("/")
    dest_root = Path(dest_root) 

    remote_files = _list_remote_files(doi)
    keys = [k for k in remote_files if k == dv_prefix or k.startswith(dv_prefix + "/")]
    if dv_prefix in keys:
        keys.remove(dv_prefix)

    if not keys:
        print(f"[download_folder] no remote files under '{dv_prefix}'")
        return []

    print(f"[download_folder] downloading {len(keys)} files from '{dv_prefix}'")
    local_paths = []
    for key in tqdm(sorted(keys), unit="file"):
        dest_path = dest_root / key
        if dest_path.exists() and not overwrite:
            local_paths.append(str(dest_path))
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        download_by_key(doi, key, dest_path)
        local_paths.append(str(dest_path))
    return local_paths
