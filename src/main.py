import argparse, json, os ,csv
import pandas as pd
import numpy as np

from generate import Generator
from classify import SceneClassifier
from synchronize import *
from curate import Curating
from select_best import Selector

# Uncomment the movies you want to generate AD for.
movies = [
    "LES_MISERABLES",
    # "IDES_OF_MARCH",
    # "HARRY_POTTER_AND_THE_GOBLET_OF_FIRE",
    # "THE_ROOMMATE",
    # "SIGNS",
    # "CHARLIE_ST_CLOUD",
    # "LEGION",
    # "HANSEL_GRETEL_WITCH_HUNTERS",
    # "BATTLE_LOS_ANGELES",
    # "HOW_DO_YOU_KNOW"
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    
    args = parser.parse_args()
    return args

def sample_video(lower, upper, sync_delay=0, fps=5, sample_size=8):
    lower = int((lower+sync_delay) * fps)
    upper = int((upper+sync_delay) * fps)
    start_shot = lower
    end_shot = upper
    if end_shot - start_shot + 1 > sample_size:
        result = np.linspace(start_shot, end_shot, sample_size, dtype=int).tolist()
    else:
        result = [i for i in range(start_shot, end_shot+1)]
    return result

def generate_AD(args):
    annotations = pd.read_csv(os.path.join(args.rootdir, 'datasets', 'MAD_eval.csv'))
    length_threshold = 2000

    examples = pd.read_csv(os.path.join(args.rootdir, 'datasets', 'MAD_train.csv'))
    examples['duration'] = examples['end'] - examples['start']
    examples['quantized_duration'] = [round(d, 1) for d in examples['duration']]

    for movie in movies:
        video_path = os.path.join(args.rootdir, 'videos', movie)
        script_path = os.path.join(args.rootdir, 'scripts', movie)
        output_path = os.path.join(args.rootdir, 'results', movie)
        with open(os.path.join(script_path, 'config.json'), 'r') as f:
            video_config = json.load(f)
        mad_id = video_config['mad_id']
        anno = annotations[annotations['movie'] == mad_id]
        
        generator = Generator(
            script_path=script_path,
            prompt_path=os.path.join(args.rootdir, 'prompts'),
            vis_root=video_path,
            api_key=args.api_key
        )

        classifier = SceneClassifier(
            video_path,
            script_path,
            os.path.join(args.rootdir, 'prompts'),
            length_threshold,
            args.api_key
        )

        selector = Selector(prompt_path=os.path.join(args.rootdir, 'prompts'), api_key=args.api_key)

        outputs = {'idx': [], 'NarrAD_without_curation': [], 'Reference': [], 'start': [], 'end': []}
        idx = 0
        for _, row in anno.iterrows():
            # Get video frames
            frames = sample_video(row['start'], row['end'], sync_delay=video_config['sync_delay'])
            noframe = False
            for frame in frames:
                image_path = os.path.join(video_path, f"frame_{str(frame).zfill(6)}.png")

                if not os.path.exists(image_path):
                    noframe = True
                    break
            if noframe:
                print(f"Frame does not exist.")
                break
                
            # Find examples with similar duration
            duration = round(row['end']-row['start'], 1)
            df = examples[examples['quantized_duration'] == duration]
            df = df.sample(n=min(20, len(df)))['text'].tolist()
            df = "\n".join(df)

            # Dialogue synchornization
            lower, upper = select_candidate(row['start'], script_path)

            # Scene recognition
            candidate = classifier.find_scene(frames, lower, upper)

            # Generate AD query
            response = generator.generate(frames, candidate, reference=df)
            response = response.json()
            contents = response["choices"]
            predictions = []
            for content in contents:
                pred = content["message"]["content"]
                predictions.append(pred)
            
            # Select best prediction
            response = selector.select(predictions, direct=True)
            response = response.json()
            contents = response["choices"]
            predictions = []
            for content in contents:
                pred = content["message"]["content"]
                predictions.append(pred)

            # Save results
            for res in predictions:
                outputs['idx'].append(str(idx))
                outputs['NarrAD_without_curation'].append(res.strip())
                outputs['Reference'].append(anno.iloc[idx]['text'].strip())
                outputs['start'].append(anno.iloc[idx]['start'])
                outputs['end'].append(anno.iloc[idx]['end'])
            
            idx += 1
        output_df = pd.DataFrame(outputs)
        output_df.to_csv(os.path.join(output_path, "output.csv"), index=False)

def curate_AD(args):
    curator = Curating(os.path.join(args.rootdir, 'prompts'), os.path.join(args.rootdir, "ad_stats.csv"), args.api_key)
    for movie in movies:
        output_path = os.path.join(args.rootdir, 'results', movie)

        # Split AD into semantic units
        units = {}
        anno = pd.read_csv(os.path.join(output_path, "output.csv"))
        for _, row in anno.iterrows():
            prediction = row['NarrAD_without_curation']
            unit = curator.split_sentence(prediction)
            units[row['idx']] = unit
        
        # Find duplicate
        duplicates = {}
        time_window = 20
        for _, row in anno.iterrows():
            i = int(row['idx'])
            current_units = units[row['idx']].split("*")
            duration = round(row['end']-row['start'], 1)
            stats = curator.duration_stats(duration)
            current_over_ratio = curator.count_syllables(row['NarrAD_without_curation']) / stats['75%']

            # Get previous AD
            prev_i = i-1
            while prev_i >= 0:
                prev_row = anno.iloc[prev_i]
                if prev_row["start"] < row["start"] - time_window:
                    break
                else:
                    prev_i -= 1
            prev_i += 1
            prev_dict = {}
            for j in range(prev_i, i):
                tmp = anno.iloc[j]
                tmp_over_ratio = curator.count_syllables(tmp["NarrAD_without_curation"]) / curator.duration_stats(round(tmp["end"]-tmp["start"],1))["75%"]
                if tmp_over_ratio > current_over_ratio:
                    continue
                tmp_units = units[j].split("*")
                tmp_index = tmp["idx"]
                for k in range(len(tmp_units)):
                    prev_dict[f"{tmp_index}-{k}"] = tmp_units[k]

            # Get next AD
            next_i = i+1
            while next_i < len(anno):
                next_row = anno.iloc[next_i]
                if next_row["end"] > row["end"] + time_window:
                    break
                else:
                    next_i += 1
            next_i -= 1
            for j in range(i+1, next_i+1):
                tmp = anno.iloc[j]
                tmp_over_ratio = curator.count_syllables(tmp["NarrAD_without_curation"]) / curator.duration_stats(round(tmp["end"]-tmp["start"],1))["75%"]
                if tmp_over_ratio > current_over_ratio:
                    continue
                tmp_units = units[j].split("*")
                tmp_index = tmp["idx"]
                for k in range(len(tmp_units)):
                    prev_dict[f"{tmp_index}-{k}"] = tmp_units[k]
            
            # Find duplicate
            tmp_duplicate = []
            for unit_idx, unit in enumerate(current_units):
                tmp = []
                for key, value in prev_dict.items():
                    duplicate = curator.find_duplicate(unit, value)
                    if duplicate == "O":
                        tmp.append(key)
                tmp_duplicate.append("&".join(tmp))
            duplicates[row['idx']] = '*'.join(tmp_duplicate)
        
        # Iterative reconstitute
        processed = []
        for _, row in anno.iterrows():
            text = row['NarrAD_without_curation']
            unit = units[row['idx']].split("*")
            duplicate = duplicates[row['idx']].split("*")
            duplicates_count = []
            for dup in duplicate:
                if dup == "":
                    duplicates_count.append(0)
                else:
                    duplicates_count.append(len(dup.split("&")))

            indexs = [i for i in range(len(unit))]
            priority_units = list(zip(duplicates_count, unit, indexs))
            priority_units.sort(key=lambda x: x[0])

            duration = round(row['end']-row['start'], 1)
            stats = curator.duration_stats(duration)
            while curator.count_syllables(text) > stats['75%']:
                if len(priority_units) >= 2:
                    priority_units.pop()
                    text = curator.reconstitute("*".join([unit for _, unit, _ in priority_units]))
                else:
                    break
            processed.append(text)

        anno['NarrAD'] = processed
        anno.to_csv(os.path.join(output_path, "output_processed.csv"), index=False)

def dialogue_synchronization(args):
    for movie in movies:
        script_path = os.path.join(args.rootdir, 'scripts', movie)
        match_lines(script_path)

if __name__ == "__main__":
    args = get_args()

    if args.task == "generate":
        generate_AD(args)
    elif args.task == "curate":
        curate_AD(args)
    elif args.task == "synchronize":
        dialogue_synchronization(args)