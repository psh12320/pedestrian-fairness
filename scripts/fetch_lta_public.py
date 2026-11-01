"""Download selected public LTA files with a strict 40 MiB total cap; no extraction."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
from datetime import datetime, timezone

BASE = 'https://datamall.lta.gov.sg/content/dam/datamall/datasets/'
FILES = {
    'road-crossings.zip': 'Geospatial/RoadCrossing_Sep2026.zip',
    'footpaths.zip': 'Geospatial/Footpath_Sep2026.zip',
    'detector-loops.zip': 'Geospatial/DetectorLoop_Sep2026.zip',
    'signal-aspects.zip': 'Geospatial/TrafficLight_Sep2026.zip',
    'traffic-count-description.pdf': 'Geospatial/Description%20for%20Trip%20Count.pdf',
    'traffic-flow-package.zip': 'TrafficRelated/TrafficFlow.zip',
}

def main():
    dest = Path('data/lta-public/2026-11-01')
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    remaining = 40 * 1024 * 1024
    for name, suffix in FILES.items():
        url = BASE + suffix
        entry = {'file': name, 'source_url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat()}
        try:
            target = dest / name
            if target.exists():
                data = target.read_bytes()
            else:
                with urllib.request.urlopen(url, timeout=45) as response:
                    data = response.read(min(remaining, 16*1024*1024) + 1)
            if len(data) > min(remaining, 16*1024*1024):
                raise ValueError('Download size cap exceeded')
            remaining -= len(data)
            if name.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    entry['members'] = [{'name': x.filename, 'compressed_bytes': x.compress_size, 'uncompressed_bytes': x.file_size} for x in archive.infolist()]
                    entry['total_uncompressed_bytes'] = sum(x.file_size for x in archive.infolist())
            elif not data.startswith(b'%PDF'):
                raise ValueError('Response is not a PDF')
            target.write_bytes(data)
            entry.update(status='downloaded', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        except Exception as error:
            entry.update(status='failed', error=str(error))
        manifest.append(entry)
        print(name, entry['status'], entry.get('bytes', entry.get('error')))
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
