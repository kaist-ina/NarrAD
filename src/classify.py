import pandas as pd
import os, requests, base64

class SceneClassifier:
    def __init__(self, video_path, script_path, prompt_path, length_threshold, api_key):
        self.vis_root = video_path
        self.script_path = script_path
        self.prompt_path = prompt_path
        self.length_threshold = length_threshold
        self.scenes = pd.read_csv(os.path.join(self.script_path, "scenes.csv"))
        with open(os.path.join(prompt_path, "prompt_classify.txt"), "r") as f:
            self.prompt = f.readlines()
        self.prompt = "".join(self.prompt)
        self.api_key = api_key

    def find_scene(self, frames, lower, upper, candidate=None):
        total_len = 0
        for i in range(lower, upper+1):
            total_len += self.get_nth_scene_length(i)
        if total_len <= self.length_threshold or "HANSEL_GRETEL_WITCH_HUNTERS" in self.script_path:
            return [i for i in range(lower, upper+1)]

        # Scene classification
        scenes = self.scenes.iloc[lower:upper+1]
        images = self.get_image(frames)
        prompt = self.prompt
        for i in range(upper-lower+1):
            prompt += f"{i+lower}: {scenes.iloc[i]['text']}\n"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": prompt
                }
                ]
            }
            ],
            "max_tokens": 300,
            "n": 1
        }
        for image in images:
            content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}"
                }
            }
            payload["messages"][0]["content"].append(content)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()
        output = response["choices"][0]["message"]["content"]
        output = output.strip().split(",")
        candidate = [int(i) for i in output]
    
        # Expansion
        window = sorted(candidate)
        total_len = self.scene_length(window)
        while total_len <= self.length_threshold:
            last_window = [i for i in window]

            stop = False
            for s in last_window:
                if not max(s-1, lower) in window:
                    window.append(max(s-1, lower))
                    window = sorted(list(set(window)))
                    if self.scene_length(window) > self.length_threshold:
                        window.remove(max(s-1, lower))
                        stop = True
                        break
                if not min(s+1, upper) in window:
                    window.append(min(s+1, upper))
                    window = sorted(list(set(window)))
                    if self.scene_length(window) > self.length_threshold:
                        window.remove(min(s+1, upper))
                        stop = True
                        break
            if stop:
                break
            total_len = self.scene_length(window)
        window = list(set(window))
        window = sorted(window)
        return window


    def get_nth_scene_length(self, scene_number):
        current_scene = 0
        scene_body = []
        in_correct_scene = False
        filename = os.path.join(self.script_path, "stage_directions.txt")

        with open(filename, 'r') as file:
            for line in file:
                if line.split(':')[0].isdigit():
                    if int(line.split(':')[0]) == scene_number:
                        in_correct_scene = True
                        current_scene = scene_number
                    else:
                        if in_correct_scene:
                            break
                        current_scene += 1
                        scene_body = []
                elif in_correct_scene and line.strip() != '':
                    scene_body.append(line.strip())
        
        scene_body = '\n'.join(scene_body)
        scene_body = scene_body.split()
        return len(scene_body)
    
    def scene_length(self, scene_numbers):
        res = 0
        for s in scene_numbers:
            res += self.get_nth_scene_length(s)
        return res
    
    def get_image(self, frame, vis_root=None):
        if isinstance(frame, list):
            frame_id = frame
        elif "-" in frame:
            start = int(frame.split("-")[0])
            end = int(frame.split("-")[1])
            frame_id = [i for i in range(start, end+1)]
        else:
            frame_id = [int(frame)]
        
        if vis_root == None:
            video_path = self.vis_root
        else:
            video_path = vis_root

        images = []
        for id in frame_id:
            image_path = os.path.join(video_path, f"frame_{str(id).zfill(6)}.png")

            with open(image_path, "rb") as image_file:
                images.append(base64.b64encode(image_file.read()).decode('utf-8'))
        return images