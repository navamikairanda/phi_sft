import json

def read_json(file):
	with open(file,'r') as f:
		config = json.load(f)
	return config

def save_json(json_data, file):
	with open(file,'w') as f:
		json.dump(json_data, f)
