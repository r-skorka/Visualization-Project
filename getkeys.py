import json

with open('config.json') as config_file:
    config = json.load(config_file)
    VISUAL_CROSSING_KEY = config["VISUAL_CROSSING_KEY"]
    IS_IT_WATER_KEY = config["IS_IT_WATER_KEY"]
    GEO_CODING_KEY = config["GEO_CODING_KEY"]