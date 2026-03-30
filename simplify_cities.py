import json
import os

def simplify_json(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data.get('data', []):
        if 'name' in item and len(item['name']) > 0:
            # Keep only the 'ar' key if it exists
            ar_name = item['name'][0].get('ar', '')
            item['name'] = [{"ar": ar_name}]
            
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Simplified: {file_path}")

if __name__ == "__main__":
    paths = [
        'model_desgin_1/libyan_cities.json',
        'model_desgin_2/libyan_cities.json'
    ]
    for p in paths:
        simplify_json(p)
