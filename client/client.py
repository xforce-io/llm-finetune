import requests
import json
import fire

def post_request(url, prompt):
    # 构建请求数据
    data = {
        'prompt': prompt
    }

    # 设置 headers，包括 Content-Type
    headers = {'Content-Type': 'application/json'}

    # 发送 POST 请求
    response = requests.post(url, json=data, headers=headers)

    # 处理响应
    if response.status_code == 200:
        result = response.json()
        # 对返回的 JSON 数据进行处理
        return result
    else:
        return None

if __name__ == '__main__':
    fire.Fire(post_request)