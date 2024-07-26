import json

def get_config(cfg_filename):
    try:
        with open(cfg_filename) as file:
            cfg = file.read()
        return json.loads(cfg.replace("\n", ""))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {cfg_filename}: {e}")
        return None