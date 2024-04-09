import requests
from time import sleep


class GPT:
    def __init__(self, sleep_time=0.5):
        self.sleep_time = sleep_time
        self.SK = 'Bearer sk-oPYA5J7cxaEqfrQuo78T77aIAACPa5XFRY3R8OKpTPqxCiSU'
        self.URL = 'https://api.chatanywhere.com.cn/v1/chat/completions'
    
    def chat_request(self, messages):
        sleep(self.sleep_time)
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.SK
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0.0
        }
        response = requests.post(self.URL, headers=headers, json=data)
        try:
            res = response.json()["choices"][0]["message"]["content"].strip()
        except:
            print(response)
            return None
        return res
        