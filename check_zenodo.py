import urllib.request
import json
url = 'https://zenodo.org/api/records/1199011'
with urllib.request.urlopen(url) as response:
    data = json.loads(response.read().decode())
for f in data.get('files', []):
    print(f"{f['key']}: {f['size']/(1024*1024):.1f} MB, {f['checksum']}")
