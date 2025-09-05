
from databricks.sdk import WorkspaceClient
from typing import Union
import os

CHUNK = 1024 * 1024  # 1MB

def _ensure_parent(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def upload(local_path: str, remote_path: str) -> None:

    w = WorkspaceClient()
    _ensure_parent(local_path)
    size = os.path.getsize(local_path)
    with open(local_path, "rb") as f:
        w.files.upload(file_path=remote_path, contents=f, overwrite=True)
    print(f"[upload] {local_path} -> {remote_path} ({size} bytes)")

def _write_bytes(dst_f, data: Union[bytes, bytearray, memoryview, str]) -> int:
    if isinstance(data, str):
        data = data.encode()
    b = bytes(data)
    dst_f.write(b)
    return len(b)

def _download_via_requests(remote_path: str, local_path: str) -> int:
    import requests
    host  = (os.environ.get("DATABRICKS_HOST") or "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN") or ""
    if not host or not token:
        raise RuntimeError("DATABRICKS_HOST/TOKEN are required for REST fallback")
    url = f"{host}/api/2.0/files/download"
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, params={"path": remote_path}, stream=True) as r:
        r.raise_for_status()
        total = 0
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                total += _write_bytes(f, chunk)
    return total

def download(remote_path: str, local_path: str) -> int:
    w = WorkspaceClient()
    _ensure_parent(local_path)

    resp = w.files.download(file_path=remote_path)

    with open(local_path, "wb") as f:
        contents = getattr(resp, "contents", None)
        if isinstance(contents, (bytes, bytearray, memoryview, str)):
            n = _write_bytes(f, contents)
            print(f"[download] {remote_path} -> {local_path} ({n} bytes) [bytes]")
            return n

        reader = getattr(contents, "read", None) if contents is not None else None
        if callable(reader):
            total = 0
            while True:
                chunk = reader(CHUNK)
                if not chunk:
                    break
                total += _write_bytes(f, chunk)
            print(f"[download] {remote_path} -> {local_path} ({total} bytes) [contents.read]")
            return total

        if contents is not None:
            try:
                it = iter(contents)
                total = 0
                for chunk in it:
                    if not chunk:
                        continue
                    total += _write_bytes(f, chunk)
                print(f"[download] {remote_path} -> {local_path} ({total} bytes) [iter(contents)]")
                return total
            except TypeError:
                pass

        resp_reader = getattr(resp, "read", None)
        if callable(resp_reader):
            total = 0
            while True:
                chunk = resp_reader(CHUNK)
                if not chunk:
                    break
                total += _write_bytes(f, chunk)
            print(f"[download] {remote_path} -> {local_path} ({total} bytes) [resp.read]")
            return total

    total = _download_via_requests(remote_path, local_path)
    print(f"[download] {remote_path} -> {local_path} ({total} bytes) [REST]")
    return total