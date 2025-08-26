import os, json, requests, sys
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

HOST  = (os.environ.get("DATABRICKS_HOST") or "").rstrip("/")
TOKEN = os.environ.get("DATABRICKS_TOKEN") or ""
NB_DIR= os.environ.get("NOTEBOOK_DIR")

def die(msg): print(msg, file=sys.stderr); sys.exit(1)
if not HOST.startswith("https://"): die("DATABRICKS_HOST is missing/invalid")
if not TOKEN.startswith("dapi"):     die("DATABRICKS_TOKEN missing/invalid (must start with dapi)")

def _replace(obj, placeholder, value):
    if value is None: return obj
    if isinstance(obj, dict):  return {k:_replace(v, placeholder, value) for k,v in obj.items()}
    if isinstance(obj, list):  return [_replace(x, placeholder, value) for x in obj]
    if isinstance(obj, str):   return obj.replace(placeholder, value)
    return obj

def create_job(json_path: str) -> int:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if NB_DIR:
        payload = _replace(payload, "<<NOTEBOOK_DIR>>", NB_DIR)

    # DEBUG: print final payload once
    print(f"\n--- PAYLOAD {json_path} ---")
    print(json.dumps(payload, indent=2))

    url = f"{HOST}/api/2.1/jobs/create"
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code >= 300:
        print(r.text)
        r.raise_for_status()
    jid = r.json().get("job_id")
    print(f"[OK] Created job {jid} from {json_path}")
    return int(jid)

if __name__ == "__main__":
    a = create_job("jobA_serverless.json")
    b = create_job("jobB_serverless.json")
    print("\nJob A id:", a)
    print("Job B id:", b)
