"""Verify and unpack the frozen workload, including one readable JSONL per batch."""

import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    archive = here / "serving_benchmark_v1.zip"
    destination = here / archive.stem
    checksum = (here / (archive.name + ".sha256")).read_text().split()[0]
    data = archive.read_bytes()
    if hashlib.sha256(data).hexdigest() != checksum:
        raise SystemExit(f"Checksum mismatch for {archive.name}")
    if destination.exists():
        raise SystemExit(
            f"{destination.name} already exists. Use a fresh directory for another copy."
        )

    with tempfile.TemporaryDirectory(dir=here) as temp:
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            bundle.extractall(temp)
        root = Path(temp) / archive.stem
        with (root / "requests.jsonl").open("rb") as source:
            records = {json.loads(line)["id"]: line for line in source}
        for batch in sorted((root / "batches").glob("*.json")):
            ids = json.loads(batch.read_text())
            with (root / (batch.stem + ".jsonl")).open("wb") as output:
                output.writelines(records[request_id] for request_id in ids)
        root.rename(destination)
    print(f"Ready: {destination}")


if __name__ == "__main__":
    main()
