import urllib.request, json, urllib.parse

query = '("large language models" OR "LLM") AND "clinical vignettes" AND "benchmark"'
url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=' + urllib.parse.quote(query) + '&format=json&resultType=core'
req = urllib.request.Request(url)
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        for item in data.get('resultList', {}).get('result', [])[:5]:
            print(f"- {item.get('title', '')}")
except Exception as e:
    print(f'Error: {e}')
