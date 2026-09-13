import sys
sys.stdout.reconfigure(encoding='utf-8')
import urllib.request, json, urllib.parse

query = 'benchmark large language models clinical vignettes hallucinations'
url = 'https://api.crossref.org/works?query=' + urllib.parse.quote(query) + '&select=title,abstract,author,published-print,URL&rows=5'
req = urllib.request.Request(url, headers={'User-Agent': 'mailto:test@example.com'})
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        for item in data['message']['items']:
            print(f"- {item.get('title', [''])[0]}")
            abstract = item.get('abstract', 'No abstract')
            print(f"  {abstract[:200]}...\n")
except Exception as e:
    print(f'Error: {e}')
