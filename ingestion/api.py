import os
import json
import requests

def fetch_data_from_api(api_url: str, save_path: str) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(api_url)
    if response.ok:
        with open(save_path, "w") as f:
            json.dump(response.json(), f)
        return save_path
    else:
        raise Exception(f"API call failed: {response.status_code}")
