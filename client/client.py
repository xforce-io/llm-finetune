import requests
import json
import fire

def post_request(url, prompt):
    # 构建请求数据
    data = {
        'prompt': prompt,
        'key1': 'value1',
        'key2': 'value2'
    }

    # 发送 POST 请求
    response = requests.post(url, json=data)

    # 处理响应
    if response.status_code == 200:
        result = response.json()
        # 对返回的 JSON 数据进行处理
        print(result)
    else:
        print('请求失败')

if __name__ == '__main__':
    fire.Fire(post_request)