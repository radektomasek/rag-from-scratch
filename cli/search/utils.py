import json

def data_read(file_name: str):
    with open(file_name, "r") as file:
        return json.load(file)