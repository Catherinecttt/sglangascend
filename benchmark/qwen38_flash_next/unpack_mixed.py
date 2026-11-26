"""Unpack readable batches and OpenAI-format inputs for the serving benchmark."""

import base64
import errno
import filecmp
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path


def write_openai_groups(root, records):
    image_urls = {}
    for path in (root / "images").glob("*.png"):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        relative_path = path.relative_to(root).as_posix()
        image_urls[relative_path] = "data:image/png;base64," + encoded

    output_tokens = json.loads((root / "summary.json").read_text())["output_tokens"]
    output_dir = root / "openai"
    output_dir.mkdir()
    for group in "ABCD":
        ids = json.loads((root / "batches" / f"{group}256.json").read_text())
        with (output_dir / f"{group}.jsonl").open("w", encoding="utf-8") as output:
            for request_id in ids:
                row = json.loads(records[request_id])
                content = [{"type": "text", "text": row["prompt"]}]
                content.extend(
                    {"type": "image_url", "image_url": {"url": image_urls[image]}}
                    for image in row["images"]
                )
                request = {
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": output_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                    "stream_options": {"include_usage": True},
                }
                output.write(json.dumps(request, separators=(",", ":")) + "\n")


def install_files(root, destination):
    try:
        root.rename(destination)
        return
    except OSError as error:
        if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise

    # An earlier unpack can already contain the readable batches. Add missing
    # files atomically, preserving any local edits and benchmark results.
    for source in root.rglob("*"):
        if not source.is_file():
            continue
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.hardlink_to(source)
        except FileExistsError:
            if not filecmp.cmp(source, target, shallow=False):
                raise SystemExit(f"Existing file differs from frozen data: {target}")


def main():
    here = Path(__file__).resolve().parent
    archive = here / "serving_benchmark_v1.zip"
    destination = here / archive.stem
    checksum = (here / (archive.name + ".sha256")).read_text().split()[0]
    data = archive.read_bytes()
    if hashlib.sha256(data).hexdigest() != checksum:
        raise SystemExit(f"Checksum mismatch for {archive.name}")

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
        write_openai_groups(root, records)
        install_files(root, destination)
    print(f"Ready: {destination}")


if __name__ == "__main__":
    main()
