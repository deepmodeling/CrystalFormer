import matplotlib.pyplot as plt
from ase.db import connect

db = connect('c2db.db')
rows = db.select('layergroup')

rows.layergroup