import json, os, requests, random

class Selector:
    def __init__(self, prompt_path, api_key):
        self.prompt_path = prompt_path
        with open(os.path.join(self.prompt_path, "prompt_select.txt"), "r") as f:
            self.prompt = f.readlines()
        self.prompt = "".join(self.prompt)
        with open(os.path.join(self.prompt_path, "prompt_select_examples.json"), "r") as f:
            self.examples = json.load(f)
        self.api_key = api_key

    def select(self, predictions, direct=False):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
            {
                "role": "system",
                "content": [
                {
                    "type": "text",
                    "text": self.prompt
                }
                ]
            }
            ],
            "max_tokens": 300,
            "n": 1
        }
        for example in self.examples:
            user_example = {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': example['cand']
                    }
                ]
            }
            assistant_example = {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': example['ref']
                    }
                ]
            }
            payload['messages'].append(user_example)
            payload['messages'].append(assistant_example)
        user_example = {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': "\n".join(predictions)
                }
            ]
        }
        payload['messages'].append(user_example)
        if direct:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            return response
        else:
            return payload
