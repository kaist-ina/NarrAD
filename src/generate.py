import requests, os, base64
from PIL import Image

class Generator:
    def __init__(self, script_path, prompt_path, vis_root, api_key):
        self.script_path = script_path
        self.prompt_path = prompt_path
        self.vis_root = vis_root
        self.api_key = api_key
        with open(os.path.join(self.prompt_path, "prompt.txt"), "r") as f:
            self.prompt = f.readlines()
        self.prompt = "".join(self.prompt)
        self.prompt = self.prompt.replace("MAX_WORD", "15")


    def generate(self, frame, scene, model="gpt-4o", reference=""):
        script = ""
        for s in scene:
            script += self.get_nth_scene(int(s))
            script += "\n"
        prompt = f"\"\"\"{script}\"\"\""

        images = self.get_image(frame)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            'model': model,
            'messages': [
                {
                    'role': 'system',
                    'content': [
                        {
                            'type': 'text',
                            'text': self.prompt + reference
                        }
                    ]
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': prompt
                        }
                    ]
                }
            ],
            'n': 10
        }
        for image in images:
            content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}",
                }
            }
            payload["messages"][-1]["content"].append(content)
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response
    
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

    def get_nth_scene(self, scene_number):
        current_scene = 0
        scene_title = None
        scene_body = []
        in_correct_scene = False
        filename = os.path.join(self.script_path, "stage_directions.txt")
        
        with open(filename, 'r') as file:
            for line in file:
                if line.split(':')[0].isdigit():
                    if int(line.split(':')[0]) == scene_number:
                        scene_title = line.split(':')[1].strip()
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
        return scene_title + "\n" + scene_body  
        
