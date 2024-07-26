import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import pandas as pd
import geopandas as gpd
import shapely

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

class Visualizer:

    DAY = 0
    # WEEK = 1 # uncomment when figure out how to implement this lol
    MONTH = 2
    YEAR = 3
    
    def __init__(self, scale, interval) -> None:
        self.data = {}
        self.weights = {}
        self.interval = interval
        self.scale = scale
        # get NY map
        # self.ny = gpd.read_file(gpd.datasets.get_path('nybb')).to_crs(epsg=4326)
        # self.ny.geometry = self.ny.geometry.map(lambda polygon: shapely.ops.transform(lambda x, y: (y, x), polygon))
        # self.ny = self.ny.boundary
        self.ny = gpd.read_file(gpd.datasets.get_path('nybb')).boundary

    def add_points(self, data, label, weight):
        self.data[label] = data.sort_values("datetime");
        self.weights[label] = weight

    def show_group(self, group):
        group = int(group)
        self.plot.clear()

        # redraw map
        self.ny.plot(ax=self.plot)

        # draw data points
        for label in self.data:
            df = self.data[label]
            df = df[df.group == group]  
            self.plot.scatter(df["latitude"], df["longitude"], label=label, alpha=self.weights[label])
        self.canvas.draw()
        self.plot.legend()
        
        # update date range
        if self.interval == self.MONTH:
            month = self.begin + group - 2
            date_string = months[month % 12] + " " + str(month // 12)
        elif self.interval == self.YEAR:
            date_string = self.begin + group - 1
        self.label.config(text=date_string)

    def launch_app(self):

        # get date range
        begin = -1
        end = pd.to_datetime(0)
        for label in self.data:
            df = self.data[label]
            val = df.iloc[0, :].datetime
            begin = min(begin, val) if begin != -1 else val
            end = max(end, df.iloc[-1, :].datetime)
        if self.interval == self.MONTH:
            begin = begin.year * 12 + begin.month
            end = end.year * 12 + end.month;
            intervals = end - begin + 1
            self.begin = begin
        elif self.interval == self.YEAR:
            begin = begin.year
            end = end.year
            intervals = end - begin + 1
            self.begin = begin
        
        # group data
        for label in self.data:
            df = self.data[label]
            if self.interval == self.MONTH:
                df.insert(3, "group", df.datetime.dt.year * 12 + df.datetime.dt.month - begin + 1)
            elif self.interval == self.YEAR:
                df.insert(3, "group", df.datetime.dt.year - begin + 1)

        # initialize
        self.window = tk.Tk()
        self.window.title("visualization")
        self.window.geometry("%dx%d" % (self.scale, self.scale))
        print("Initiliazed app with scale %d and interval %d" % (self.scale, self.interval))

        # draw ui elements
        if self.interval == self.MONTH:
            begin -= 1
            date_string = months[begin % 12] + " " + str(begin // 12)
        elif self.interval == self.YEAR:
            date_string = begin
        self.label = tk.Label(master=self.window, text=date_string, font=("Arial", 25))
        self.label.pack()
        w = tk.Scale(master=self.window, from_=1, to=intervals, orient="horizontal", length=900, command=self.show_group)
        w.pack()

        fig = plt.figure(figsize=(self.scale / 100, self.scale / 100), dpi=100)
        self.plot = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(figure=fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()   

        self.show_group(1)

        # run app
        self.window.mainloop()