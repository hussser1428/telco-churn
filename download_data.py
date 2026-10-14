"""Download the public CSV without requiring Kaggle credentials."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
KAGGLE_PAGE = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
SOURCES = [
    ("Kaggle / BlastChar", "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn", True),
    ("IBM public sample repository", "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv", False),
]


def download():
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if DATA.exists():
        print(f"Using existing file: {DATA}")
        return DATA
    failures = []
    for publisher, url, zipped in SOURCES:
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=45) as response:
                content = response.read()
            if zipped:
                with ZipFile(BytesIO(content)) as archive:
                    names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                    if len(names) != 1:
                        raise ValueError(f"Expected one CSV, received {names}")
                    content = archive.read(names[0])
            if not content.decode("utf-8-sig").startswith("customerID,"):
                raise ValueError("Response is not the expected Telco CSV")
            DATA.write_bytes(content)
            metadata = {
                "dataset_page": KAGGLE_PAGE, "publisher": publisher, "download_url": url,
                "sha256": sha256(content).hexdigest(), "bytes": len(content),
                "fallback_failures": failures,
                "description": "Public Telco Customer Churn sample for learning; no claim of live customer data.",
            }
            (DATA.parent / "source.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Downloaded {len(content):,} bytes from {publisher}")
            return DATA
        except Exception as exc:
            failures.append(f"{publisher}: {type(exc).__name__}: {exc}")
            print(failures[-1])
    raise RuntimeError("Unable to download dataset. " + " | ".join(failures))


if __name__ == "__main__":
    download()
