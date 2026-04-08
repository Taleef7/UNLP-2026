"""
Upload Kaggle dataset using KGAT Bearer auth (KAGGLE_API_TOKEN env var).
Usage: python3 kaggle_upload_bearer.py
"""
import json, os, sys, zipfile, time, io
import requests

# ── Credentials ────────────────────────────────────────────────────────────────
with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f:
    creds = json.load(f)
# KAGGLE_API_TOKEN triggers Bearer auth in kagglesdk
os.environ['KAGGLE_API_TOKEN'] = creds['key']
USER = creds['username']

from kagglesdk import KaggleClient
from kagglesdk.blobs.types.blob_api_service import ApiStartBlobUploadRequest, ApiBlobType
from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest, ApiDatasetNewFile

client = KaggleClient(username=USER, password=creds['key'])
print(f'Authenticated as: {USER}')

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR = '/scratch/gilbreth/tamst01/unlp2026/kaggle_datasets/qwen3-reranker-0-6b'
DATASET_SLUG = 'qwen3-reranker-0-6b'
TITLE = 'qwen3-reranker-0-6b'
ZIP_PATH = '/tmp/qwen3-reranker-0-6b.zip'

# ── Step 0: Zip ────────────────────────────────────────────────────────────────
if os.path.exists(ZIP_PATH):
    zip_size = os.path.getsize(ZIP_PATH)
    print(f'Using existing zip: {zip_size/1e6:.1f} MB at {ZIP_PATH}')
else:
    print(f'Zipping {DATASET_DIR}...')
    skip = {'dataset-metadata.json', '.gitattributes'}
    files = [f for f in os.listdir(DATASET_DIR) if f not in skip]
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in files:
            fpath = os.path.join(DATASET_DIR, fname)
            if os.path.isfile(fpath):
                print(f'  + {fname} ({os.path.getsize(fpath)/1e6:.1f} MB)')
                zf.write(fpath, fname)
    zip_size = os.path.getsize(ZIP_PATH)
    print(f'Zip: {zip_size/1e6:.1f} MB')

# ── Step 1: Get GCS upload URL ─────────────────────────────────────────────────
print('\n[1/3] Requesting blob upload URL...')
req = ApiStartBlobUploadRequest()
req.type    = ApiBlobType.DATASET
req.name    = f'{DATASET_SLUG}.zip'
req.content_type = 'application/zip'
req.content_length = zip_size
req.last_modified_epoch_seconds = int(os.path.getmtime(ZIP_PATH))

blob_resp = client.blobs.blob_api_client.start_blob_upload(req)
print(f'GCS URL: {blob_resp.create_url[:80]}...')
print(f'Token:   {blob_resp.token[:30]}...')

# ── Step 2: Upload to GCS ──────────────────────────────────────────────────────
print(f'\n[2/3] Uploading {zip_size/1e6:.0f} MB to GCS...')
t0 = time.time()
with open(ZIP_PATH, 'rb') as f:
    r = requests.put(
        blob_resp.create_url,
        data=f,
        headers={'Content-Type': 'application/zip', 'Content-Length': str(zip_size)},
        timeout=3600
    )
elapsed = time.time() - t0
speed = zip_size / elapsed / 1e6
print(f'GCS upload: {r.status_code} ({elapsed/60:.1f} min, {speed:.1f} MB/s)')
if r.status_code not in (200, 201):
    print(f'Error: {r.text[:300]}')
    sys.exit(1)

# ── Step 3: Create dataset on Kaggle ──────────────────────────────────────────
print('\n[3/3] Creating Kaggle dataset...')
from kagglesdk.datasets.services.dataset_api_service import DatasetApiClient

ds_client = client.datasets.dataset_api_client

# Build create request
create_req = ApiCreateDatasetRequest()
create_req.owner_slug = USER
create_req.title = TITLE
create_req.is_private = True
create_req.license_name = 'Apache 2.0'
# Add the uploaded blob as a file
new_file = ApiDatasetNewFile()
new_file.token = blob_resp.token
create_req.files = [new_file]

create_resp = ds_client.create_dataset(create_req)
print(f'Status: {create_resp.status}')
print(f'Error:  {create_resp.error}')
print(f'URL:    {create_resp.url}')

if create_resp.url and 'kaggle.com' in create_resp.url:
    print(f'\nDataset created: {create_resp.url}')
    print(f'Add to kernel as: taleeftamsal/{DATASET_SLUG}')
else:
    print(f'\nUnexpected response. Full: status={create_resp.status!r} error={create_resp.error!r}')
