import pandas as pd
import os, requests, json
from tqdm import tqdm


def count_syllables(text):
    vowels = "aeiouy"  # Consider 'y' as a vowel in certain contexts
    text = text.lower()  # Convert the text to lowercase for consistency
    syllable_count = 0
    previous_char_was_vowel = False

    for char in text:
        if char in vowels:
            # Only count as a new syllable if the previous character was not a vowel
            if not previous_char_was_vowel:
                syllable_count += 1
            previous_char_was_vowel = True
        else:
            previous_char_was_vowel = False

    # Handling the rule where a final silent 'e' is not counted
    if text.endswith('e') and syllable_count > 1:
        syllable_count -= 1

    return syllable_count

class Curating:
    def __init__(self, prompt_path, stats_path, api_key):
        self.prompt_path = prompt_path
        self.stats = pd.read_csv(stats_path)
        print(len(self.stats))

        self.prompt_split = os.path.join(self.prompt_path, "prompt_split.txt")
        with open(self.prompt_split, "r") as f:
            self.prompt_split = f.readlines()
        self.prompt_split = "".join(self.prompt_split)

        self.prompt_reconstitute = os.path.join(self.prompt_path, "prompt_reconstitute.txt")
        with open(self.prompt_reconstitute, "r") as f:
            self.prompt_reconstitute = f.readlines()
        self.prompt_reconstitute = "".join(self.prompt_reconstitute)

        self.prompt_duplicate = os.path.join(self.prompt_path, "prompt_duplicate.txt")
        with open(self.prompt_duplicate, "r") as f:
            self.prompt_duplicate = f.readlines()
        self.prompt_duplicate = "".join(self.prompt_duplicate)

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def duration_stats(self, duration):
        duration_stats = self.stats[self.stats['quantized_duration'] == duration].iloc[0]
        return duration_stats
    
    def count_syllables(self, text):
        vowels = "aeiouy"  # Consider 'y' as a vowel in certain contexts
        text = text.lower()  # Convert the text to lowercase for consistency
        syllable_count = 0
        previous_char_was_vowel = False

        for char in text:
            if char in vowels:
                # Only count as a new syllable if the previous character was not a vowel
                if not previous_char_was_vowel:
                    syllable_count += 1
                previous_char_was_vowel = True
            else:
                previous_char_was_vowel = False

        # Handling the rule where a final silent 'e' is not counted
        if text.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return syllable_count
    
    def split_sentence(self, sentence):
        prompt = self.prompt_split
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
            {
                "role": "system",
                "content": [
                {
                    "type": "text",
                    "text": prompt
                }
                ]
            },
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "The house was once part of a plantation and it was the home of Josiah Henson, a slave who escaped to Canada in 1830 and wrote the story of his life."
                }
                ]
            },
            {
                "role": "assistant",
                "content": [
                {
                    "type": "text",
                    "text": "The house was once part of a plantation.*It was the home of Josiah Henson.*Josiah Henson was a slave.*This slave escaped to Canada.*This was in 1830.*This slave wrote the story of his life."
                }
                ]
            },
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "The amabassadors arrival has not been announced and he flies in complete secrecy, the official said."
                }
                ]
            },
            {
                "role": "assistant",
                "content": [
                {
                    "type": "text",
                    "text": "The ambassadors arrival has not been announced.*He flies in complete secrecy.*This is what the official said."
                }
                ]
            },
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": sentence
                }
                ]
            },
            ],
            "max_tokens": 300,
            "n": 1
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        try:
            if response.ok:
                response = response.json()
                content = response['choices'][0]['message']['content']
                return content
            else:
                return None
        except:
            print(response)
            return None

    def find_duplicate(self, target, reference):
        """
        Find sentences with same meaning with target
        target: single semantic unit
        reference: another semantic unit
        """
        prompt = f'{target}*{reference}'
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
            {
                "role": "system",
                "content": [
                {
                    "type": "text",
                    "text": self.prompt_duplicate
                }
                ]
            },
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": prompt
                }
                ]
            },
            ],
            "max_tokens": 300,
            "n": 1
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        try:
            if response.ok:
                response = response.json()
                content = response['choices'][0]['message']['content']
                return content
            else:
                return None
        except:
            print(response)
            return None

    def reconstitute(self, units):
        if units == "":
            return ""
        prompt = self.prompt_reconstitute
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
            {
                "role": "system",
                "content": [
                {
                    "type": "text",
                    "text": prompt
                }
                ]
            },
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "Charlie watches*This is at dusk"
                }
                ]
            },
            {
                "role": "assistant",
                "content": [
                {
                    "type": "text",
                    "text": "Charlie watches at dusk."
                }
                ]
            },
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "Sara is in bed with a hangover*Sara takes aspirin.*This aspirin is handed to her by Rebecca."
                }
                ]
            },
            {
                "role": "assistant",
                "content": [
                {
                    "type": "text",
                    "text": "Sara, in bed with a hangover, takes aspirin handed to her by Rebecca."
                }
                ]
            },
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": units
                }
                ]
            },
            ],
            "max_tokens": 300,
            "n": 1
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        try:
            if response.ok:
                response = response.json()
                content = response['choices'][0]['message']['content']
                content = content.split("Output: ")[-1]
                return content
            else:
                return None
        except:
            print(response)
        return None


