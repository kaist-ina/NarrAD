import pandas as pd
import os, csv
from rapidfuzz import fuzz

def select_candidate(start_time, script_path):
    if "HANSEL_GRETEL_WITCH_HUNTERS" in script_path:
        return 0, 0
    scene_boundary = pd.read_csv(os.path.join(script_path, "boundary.csv"))
    scene_tags = pd.read_csv(os.path.join(script_path, "scenes.csv"))
    last_scene = len(scene_tags)-1

    # Select Candidate
    for i, row in scene_boundary.iterrows():
        if start_time > row["start"]:
            continue
        else:
            break

    scene_upper_bound = scene_boundary.iloc[i]["scene"]
    scene_lower_bound = scene_boundary.iloc[max(0, i-1)]["scene"]

    if scene_boundary.iloc[len(scene_boundary)-1]["start"] < start_time:
        scene_upper_bound = last_scene
        scene_lower_bound = scene_boundary.iloc[i]["scene"]
    if i == 0:
        scene_upper_bound = scene_boundary.iloc[i]["scene"]
        scene_lower_bound = 0

    return scene_lower_bound, scene_upper_bound

def rank_sentences(query, sentences):
    similarities = [(sentence, fuzz.token_sort_ratio(query, sentence), scene) for sentence, scene in sentences]
    sorted_sentences = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_sentences

def match_lines(script_path):
    transcribes = os.path.join(script_path, "transcribe.csv")
    if not os.path.exists(transcribes):
        print("No transcription")
        return None
    transcribes = pd.read_csv(transcribes)

    lines = os.path.join(script_path, "lines.csv")
    if not os.path.exists(lines):
        print("No lines")
        return None
    lines = pd.read_csv(lines)

    target = []
    for _, row in lines.iterrows():
        target.append((row["text"].strip(), row["scene"]))

    threshold = 50
    total = 0
    prev_scene = -1
    result = {"query": [], "sentence": [], "scene": [], "start": [], "end": [], "score": []}
    for i, row in transcribes.iterrows():
        # Skip low confidence transcript
        if row["confidence"] <= 0.7:
            continue
        
        # Skip too short sentence
        if len(row["transcript"].strip().split(" ")) <= 4:
            continue
        
        query = row["transcript"].strip()
        ranked_sentences = rank_sentences(query, target)
        sentence, score, scene = ranked_sentences[0]
        
        # Skip low matching score
        if score <= 67:
            continue
        
        # Skip multiple matched lines
        sentence2, score2, scene2 = ranked_sentences[1]
        if score2 >= 64:
            continue

        # Skip irregularity
        if abs(scene - prev_scene) >= threshold or scene < prev_scene:
            continue

        result["query"].append(query)
        result["sentence"].append(sentence)
        result["scene"].append(scene)
        result["start"].append(row["start"])
        result["end"].append(row["end"])
        result["score"].append(score)
        total += 1
        prev_scene = scene

    with open(os.path.join(script_path, "boundary.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "start", "end", "score", "query", "script"])
        for i in range(len(result["start"])):
            next_i = min(i+1, len(result["start"])-1)
            if result["scene"][i] == result["scene"][next_i] and result["start"][next_i] - result["start"][i] >= 240:
                continue
            writer.writerow([result["scene"][i], result["start"][i], result["end"][i], result["score"][i], result["query"][i], result["sentence"][i]])