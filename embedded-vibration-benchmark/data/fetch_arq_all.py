"""Fetch all three pinned ARQ-CRISP objects with upstream Git-LFS verification."""
from pathlib import Path
import hashlib
import json
import urllib.request

COMMIT = 'a536ee3301af0566218782b00de334b6a6630979'
FILES = (
 ('brush', 'slipDataset_brush_tactile.h5', 'a2aef56bf4cfc5e2d17ae09772295cb207c733e3a73059913939ba5a7a1e2139', 41105856),
 ('screwDriver', 'slipDataset_screwDriver_tactile.h5', 'c8124b295a6faf42c750c75e8dcd49dcc45289e06fa7cb4047476af2f570309a', 82416420),
 ('SpoolSolder', 'slipDataset_SpoolSolder_tactile.h5', 'a8c4f14e71c0b4d7da738997c965e745f75123fea7f5c6d338810ba4adfdeeff', 142749496),
)
BASE = f'https://raw.githubusercontent.com/ARQ-CRISP/slip_detection_dataset_2021/{COMMIT}/'


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    target = root/'public'/'ARQ_CRISP_2021'
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for object_name, name, expected, size in FILES:
        pointer = urllib.request.urlopen(BASE+'data/'+name, timeout=30).read().decode()
        if f'oid sha256:{expected}' not in pointer or f'size {size}' not in pointer:
            raise ValueError(f'Upstream pinned LFS pointer mismatch: {name}')
        url = f'https://media.githubusercontent.com/media/ARQ-CRISP/slip_detection_dataset_2021/{COMMIT}/data/{name}'
        path = target/name
        if not path.exists() or checksum(path) != expected:
            temporary = path.with_suffix('.download')
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as out:
                while chunk := response.read(1024*1024):
                    out.write(chunk)
            if temporary.stat().st_size != size or checksum(temporary) != expected:
                temporary.unlink()
                raise ValueError(f'Download mismatch: {name}')
            temporary.replace(path)
        records.append({'object': object_name, 'filename': name, 'bytes': size,
                        'sha256': expected, 'source_url': url, 'verified': True})
        print(f'Verified {object_name}: {size} bytes', flush=True)
    for remote, local in [('LICENSE','LICENSE'), ('README.md','UPSTREAM_README.md'),
                          ('data/README.md','UPSTREAM_DATA_README.md')]:
        (target/local).write_bytes(urllib.request.urlopen(BASE+remote, timeout=30).read())
    manifest = {'dataset': 'ARQ-CRISP slip_detection_dataset_2021', 'source_commit': COMMIT,
        'license': 'BSD-3-Clause repository license; retained data/public/ARQ_CRISP_2021/LICENSE',
        'evidence_type': 'public_hardware_recording_not_our_hardware', 'files': records}
    (root/'arq_all_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
