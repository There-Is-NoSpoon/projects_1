from processors import *
from visualizer import Visualizer
import matplotlib.pyplot as plt

TIME_INTERVAL = Visualizer.YEAR # time interval in days
SCALE = 1000 # size of map in pixels

vs = Visualizer(SCALE, TIME_INTERVAL)
vs.add_points(get_shooting_data("data/Shooting_data.csv"), "shootings", 1)
vs.add_points(get_rat_data("data/rodent_edited3.csv"), "rats", 0.01)
vs.launch_app()
