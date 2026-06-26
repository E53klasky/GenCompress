import os
import json

def save_json(json_pth, data, mode = "update"):
    # Load the existing JSON data
    if os.path.exists(json_pth):
        with open(json_pth, 'r') as json_file:
            existing_data = json.load(json_file)
        if mode == "update":
            existing_data.update(data)
            data = existing_data
        elif mode == "cat":
            for key in data:
                if key in existing_data:
                    existing_data[key] += data[key]
                else:
                    existing_data[key] = data[key]
            data = existing_data
    # Save the updated data back to the JSON file
    with open(json_pth, 'w') as json_file:
        json.dump(data, json_file, indent=4)
