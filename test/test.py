import json
from client.client import post_request

kUrl = ""
kPathPclue = "benchmark/pclue/pclue.jsonl"

def test(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            prompt = json_obj["instruction"]
            resp = post_request(kUrl, prompt)
            print(resp)

if __name__ == "__main__" :
    test(kPathPclue)