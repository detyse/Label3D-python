# function about skeletons read and write
from dataclasses import dataclass
import json

# class or function

@dataclass
class Skeleton:
    def __init__(self, json_file):
        # self.name = name
        # self.joint = joint
        # self.connection = connection
        # self.color = color
        self.json_file = json_file
        self.color, self.joint_names, self.joints_idx = self.read_json(json_file)

    
    def __post_init__(self, json_file):
        pass
    