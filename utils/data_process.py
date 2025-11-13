import json
import os

def read_jsonl(file):
    with open(file,'r')as f:
        data=[json.loads(line) for line in f]
    return data

def read_json_data(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        data=json.load(f)
    return data

def write_json_from_dict(data_list,filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath,'w',encoding='utf-8') as f:
        json.dump(data_list,f,ensure_ascii=False,indent=4)

