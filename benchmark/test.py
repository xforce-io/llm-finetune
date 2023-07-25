import json
from client.client import post_request

kUrl = "http://127.0.0.1:5000/v1/completions"
kPathPclue = "benchmark/pclue/pclue.jsonl"
kBeacon = "答案"

def test(filepath):
    cntRight = 0
    all = 0
    with open(filepath, 'r') as file:
        for line in file:
            all += 1

            json_obj = json.loads(line)
            prompt = json_obj["instruction"]
            resp = post_request(kUrl, prompt)
            text = resp["choices"][0]["text"]

            idx = text.find(kBeacon)
            if idx < 0 :
                continue
            
            processed = text[idx + len(kBeacon):].replace("：", "").replace("<unk>", "").strip()
            answer = processed[:processed.find("<")]
            expected = json_obj["output"]
            if answer == expected:
                cntRight += 1
            else :
                print(f"expected[{expected}] actual[{answer}] wrong for '{prompt}' ")
    print(f"overall: right[{cntRight}] ratio[{cntRight*1.0/all}]")

if __name__ == "__main__" :
    test(kPathPclue)