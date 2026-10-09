"""Retrieve the pinned public brush recording, retaining upstream notices."""
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = 'a536ee3301af0566218782b00de334b6a6630979'
NAME = 'slipDataset_brush_tactile.h5'
SHA256 = 'a2aef56bf4cfc5e2d17ae09772295cb207c733e3a73059913939ba5a7a1e2139'
BASE = f'https://raw.githubusercontent.com/ARQ-CRISP/slip_detection_dataset_2021/{COMMIT}/'
URL = f'https://media.githubusercontent.com/media/ARQ-CRISP/slip_detection_dataset_2021/{COMMIT}/data/{NAME}'


def main():
    target = Path(__file__).resolve().parent / 'public' / 'ARQ_CRISP_2021'
    target.mkdir(parents=True, exist_ok=True)
    path = target / NAME
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        temporary = path.with_suffix('.download')
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            temporary.unlink()
            raise ValueError('Downloaded data checksum does not match pinned Git-LFS SHA256')
        temporary.replace(path)
    for remote, local in [('LICENSE', 'LICENSE'), ('README.md', 'UPSTREAM_README.md'),
                          ('data/README.md', 'UPSTREAM_DATA_README.md')]:
        (target/local).write_bytes(urllib.request.urlopen(BASE+remote, timeout=30).read())
    (target/'provenance.json').write_text(json.dumps({
        'dataset': 'ARQ-CRISP slip_detection_dataset_2021',
        'source_commit': COMMIT, 'source_url': URL, 'filename': NAME,
        'bytes': path.stat().st_size, 'sha256': SHA256,
        'license': 'BSD-3-Clause repository license; see LICENSE',
        'evidence_type': 'public_hardware_recording_not_our_hardware'}, indent=2)+'\n')
    print(f'Verified {path}: {path.stat().st_size} bytes')


if __name__ == '__main__':
    main()
