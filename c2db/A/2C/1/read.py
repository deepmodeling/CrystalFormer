import re

def lattice_from_file(file_path):
    f = open(file_path, "r")
    lines = f.readlines()
    lattice_str = re.search(r'Lattice="([^"]+)"', lines[1]).group(1)
    lattice_list = [eval(v) for v in lattice_str]