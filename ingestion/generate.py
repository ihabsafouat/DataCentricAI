import os
import json
import random

def generate_synthetic_pet_moods(save_path: str, num_samples: int = 100) -> str:
    moods = ['happy', 'sad', 'angry', 'relaxed']
    samples = [{"pet_id": i, "mood": random.choice(moods)} for i in range(num_samples)]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(samples, f, indent=2)
    return save_path
