import matplotlib.pyplot as plt
import numpy as np

def plot_circular_eeg(channel_power, freqs, title=None):
	# channel_power: numpy array of shape (numSamples,) for a specific channel
	numSamples = len(channel_power)
	max_freq = max(freqs)

	# Plots radial frequency/power figure.
	## TODO: Replace scatterplot with different plot object to better visualize the data. A standard plot(...) should work.
	# r = 2 * max_power
	r = 2 * channel_power
	theta = 2 * np.pi * (freqs / max_freq)
	# area = 200 * r ** 2
	colors = theta
	fig = plt.figure()
	# ax = fig.add_subplot(111, projection='polar')
	ax = fig.add_subplot(111, polar=True)
	# c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
	# c = ax.scatter(theta, r, c=colors, cmap='hsv', alpha=0.75)
	c = ax.plot(theta, r, alpha=0.75)

	# render it like a donut instead of a circle: THIS ISN'T WORKING
	# ax.set_rorigin(-250)
	# ax.set_theta_zero_location('W', offset=10)
	# Confine to a sector (pizza slice):
	# ax.set_thetamin(45)
	# ax.set_thetamax(135)

	# currAx = fig.add_subplot(14,1,(aChannelIndex+1))
	# currAx.psd(ch)

	if title:
		ax.set_title(title)



def plot_linear_eeg(channel_power, freqs, title=None):
	# channel_power: numpy array of shape (numSamples,) for a specific channel
	numSamples = len(channel_power)
	max_freq = max(freqs)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	# c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
	# c = ax.scatter(theta, r, c=colors, cmap='hsv', alpha=0.75)
	c = ax.plot(freqs, channel_power, alpha=0.75)
	ax.set_ylabel('Power')
	ax.set_xlabel('Frequency [Hz]')
	if title:
		ax.set_title(title)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""

import datetime

# numpy for calculations
import numpy as np

# pandas for data manipulation
import pandas as pd
from pandas import DataFrame
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# import prophet

# Plotting and Visualization
# Matplotlib for plotting
import matplotlib
matplotlib.rcParams['timezone'] = 'US/Eastern'
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colors as mcolors

from matplotlib.widgets import Slider, Button, SpanSelector, Cursor
import mplcursors

# Used for radial clock plot heatmap.
from mpl_toolkits.mplot3d import Axes3D

from pandas.plotting import autocorrelation_plot

import pylab as lab  # for plotting commands
# from pylab import plot  # for plotting commands

import seaborn as sns
from pathlib import Path
from phoofflineeeganalysis.tzinfo_examples import Eastern


class VisHelpers:
	""" Visualization Helpers
	
	from phoofflineeeganalysis.EegVisualization import VisHelpers
	 
	"""
	common_time_zone = Eastern


	hours_of_day_arr = ['Mid', '1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM', '7 AM','8 AM','9 AM','10 AM','11 AM','Noon', '1 PM', '2 PM', '3 PM', '4 PM',  '5 PM', '6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM' ]
	days_arr = ["Mon","Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
	month_names_arr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

	# Plots the times in the dataframe df on a circular clock (as a function of the time-of-day)
	@classmethod
	def plot_clock_values(cls, df, hour_col_name: str='hour', exportDir = 'Results', name = None, exportFormat='svg', use_improved_method=False):
		""" 
			from dose_analysis_python.Visualization.Visualization import plot_clock_values

		"""
		figure_name = "cummulative_hour_clock"

		if name:
			figure_name = name + " - " + figure_name

			# Verify the export directory exists
		exportPath = Path(exportDir)
		if ~exportPath.exists():
			# Try to create the path
			exportPath.mkdir(exist_ok=True)

		# Add the export directory to build the path
		figure_path = exportPath.resolve() / figure_name
		figure_name = str(figure_path)

		# Add the extension to path
		if exportFormat:
			figure_name = figure_name + "." + exportFormat

		figure = plt.figure()

		if (use_improved_method):
			# A new format added on 5-1-2020 that makes a proper heatmap
			# Inspired by https://stackoverflow.com/questions/50087868/how-do-i-create-radial-heatmap-in-matplotlib

			values_per_hour = df.groupby([df['hour']]).y.count()

			ax = Axes3D(figure)

			n = 24 # The number of circumference-divisions
			m = 1
			radial_divisions = np.linspace(0, 10, m) # The number of division that are cut along the growing radius.
			a = np.linspace(0, 2 * np.pi, n)
			r, th = np.meshgrid(radial_divisions, a)

			# z = np.random.uniform(-1, 1, (n,m))
			z = values_per_hour

			plt.subplot(projection="polar")

			plt.pcolormesh(th, r, z, cmap = 'inferno')

			plt.plot(a, r, ls='none', color = 'k') 
			plt.grid()
			plt.colorbar()

			# plt.savefig('a.png')
			# plt.show()

		else:

			angles = df.time_angle_radians.values
			# print(type(angles))
			# print(angles.shape)
			# print(len(angles))
			# print(angles)

			#ax = plt.subplot(111, polar=True)
			ax = figure.add_subplot(1, 1, 1, polar=True)

			#ax.scatter(angles, np.ones(100)*1)
			# ax.scatter(angles, np.ones(100)*1, marker='_', s=20)
			#ax.bar(angles, np.full(100, 0.9), width=0.1, bottom=0.0, color='r', linewidth=0)
			# Or, to make the bars look more like ticks, you could set bottom=0.89:
			ax.bar(angles, np.full(len(angles), 0.9), width=0.05, bottom=0.89, color='r', linewidth=0, alpha=0.2)

			# suppress the radial labels
			plt.setp(ax.get_yticklabels(), visible=False)

			# set the circumference labels
			ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
			ax.set_xticklabels(range(24))

			# make the labels go clockwise
			ax.set_theta_direction(-1)

			# place 0 at the top
			ax.set_theta_offset(np.pi/2.0)

			# plt.grid('off')

			# put the points on the circumference
			plt.ylim(0,1)

		# Only export (save) the figure to file if the exportFormat isn't None
		if exportFormat:
			figure.savefig(figure_name, format=exportFormat, dpi=1200)


	# plots a day-by-day, hour-by-hour heat map similar to the one displayed for commits on github.
	@classmethod
	def plot_github_style_heatmap(cls, df, exportDir = 'Results', name = None, exportFormat='svg'):
		figure_name = "each_date_by_hour_heatmap"

		if name:
			figure_name = name + " - " + figure_name

		# Verify the export directory exists
		exportPath = Path(exportDir)
		if ~exportPath.exists():
			# Try to create the path
			exportPath.mkdir(exist_ok=True)

		# Add the export directory to build the path
		figure_path = exportPath.resolve() / figure_name
		figure_name = str(figure_path)

		# Add the extension to path
		if exportFormat:
			figure_name = figure_name + "." + exportFormat

		figure = plt.figure(figsize = (28,5))
		axis = figure.add_subplot(1, 1, 1)

		# Getting unique values after grouping by hour and date
		grouped = df.groupby(["hour", "day_date"])
		#print(grouped["y"].sum())
		#df_new = grouped["y"].size() # Here we can either use .size(), which uses the number of doses in that (date, hour) segment
		df_new = grouped["y"].sum() # OR we can use .sum() which returns total AMPH in that segment
		df_new = df_new.reset_index(name="count")
		#print(df_new)

		# Pivot the dataframe to create a [hour x day_date] matrix containing counts
		sns.heatmap(df_new.pivot("hour", "day_date", "count"), annot=False, cmap="PuBuGn", ax = axis)

		axis.set_xlabel("Date")
		axis.set_ylabel("Hour of Day")

		plt.tight_layout()

		# Only export (save) the figure to file if the exportFormat isn't None
		if exportFormat:
			figure.savefig(figure_name, format=exportFormat, dpi=1200)


	# Displays a heatmap with day of the week on the x-axis and hour of day on the y-axis. It sums over all records provided.
	@classmethod
	def plot_github_style_dayOfWeek_hour_heatmap(cls, df, exportDir = 'Results', name = None, exportFormat='svg'):
		figure_name = "day_of_week_by_hour_heatmap"

		if name:
			figure_name = name + " - " + figure_name

			# Verify the export directory exists
		exportPath = Path(exportDir)
		if ~exportPath.exists():
			# Try to create the path
			exportPath.mkdir(exist_ok=True)

		# Add the export directory to build the path
		figure_path = exportPath.resolve() / figure_name
		figure_name = str(figure_path)

		# Add the extension to path
		if exportFormat:
			figure_name = figure_name + "." + exportFormat

		figure = plt.figure()
		axis = figure.add_subplot(1, 1, 1)

		df_heat = df.groupby(["hour", "day_of_week"])["y"].sum().reset_index()
		df_heat2 = df_heat.pivot("hour", "day_of_week", "y")

		sns.heatmap(df_heat2[0:], cbar_kws={'label': 'Dose [mg]', 'format': '%d[mg]'}, cmap="YlGnBu", ax = axis)
		axis.set_xticklabels(days_arr)
		#plt.xticks(rotation=90)
		#hm.tick_params(labelsize=8)

		axis.set_xlabel("Day of Week")
		axis.set_ylabel("Hour of Day")

		# Only export (save) the figure to file if the exportFormat isn't None
		if exportFormat:
			figure.savefig(figure_name, format=exportFormat, dpi=1200)

	@classmethod
	def plot_month_hour_heatmap(cls, df, exportDir = 'Results', name = None, exportFormat='svg'):
		figure_name = "each_date_by_month_hour_heatmap"

		if name:
			figure_name = name + " - " + figure_name

		# Verify the export directory exists
		exportPath = Path(exportDir)
		if ~exportPath.exists():
			# Try to create the path
			exportPath.mkdir(exist_ok=True)

		# Add the export directory to build the path
		figure_path = exportPath.resolve() / figure_name
		figure_name = str(figure_path)

		# Add the extension to path
		if exportFormat:
			figure_name = figure_name + "." + exportFormat

		figure = plt.figure(figsize = (28,5))
		axis = figure.add_subplot(1, 1, 1)

		# Getting unique values after grouping by hour and date
		grouped = df.groupby(["hour", "month"])
		#print(grouped["y"].sum())
		#df_new = grouped["y"].size() # Here we can either use .size(), which uses the number of doses in that (date, hour) segment
		df_new = grouped["y"].sum() # OR we can use .sum() which returns total AMPH in that segment
		df_new = df_new.reset_index(name="count")
		#print(df_new)

		# Pivot the dataframe to create a [hour x day_date] matrix containing counts
		sns.heatmap(df_new.pivot("hour", "month", "count"), annot=False, xticklabels=month_names_arr, yticklabels=hours_of_day_arr, cmap="PuBuGn", ax = axis)

		axis.set_xlabel("Month")
		axis.set_ylabel("Hour of Day")


		# Only export (save) the figure to file if the exportFormat isn't None
		if exportFormat:
			figure.savefig(figure_name, format=exportFormat, dpi=1200)



	@classmethod
	def plot_hourOfDay_Heatmap(cls, df, exportDir = 'Results', name = None, exportFormat='svg'):
		figure_name = "hourOfDay_heatmap"

		if name:
			figure_name = name + " - " + figure_name

		# Verify the export directory exists
		exportPath = Path(exportDir)
		if ~exportPath.exists():
			# Try to create the path
			exportPath.mkdir(exist_ok=True)

		# Add the export directory to build the path
		figure_path = exportPath.resolve() / figure_name
		figure_name = str(figure_path)

		# Add the extension to path
		if exportFormat:
			figure_name = figure_name + "." + exportFormat

		figure = plt.figure()
		axis = figure.add_subplot(1, 2, 1)

		# Doses per hour
		values_per_hour = df.groupby([df.hour]).y.size()
	#     print('Values per hour: {}'.format(values_per_hour))
		values_per_hour_array = values_per_hour.values.reshape((values_per_hour.size,1))

		sns.heatmap(values_per_hour_array[0:], xticklabels=False, yticklabels=hours_of_day_arr, cbar_kws={'label': 'Recorded Doses [count]', 'format': '%d'}, cmap="YlGnBu", ax = axis)
		axis.set_ylabel("Hour of Day")
		plt.yticks(rotation=45)


		#
		axis = figure.add_subplot(1, 2, 2)

		values_per_hour = df.groupby([df.hour]).y.sum()
	#     print('Values per hour: {}'.format(values_per_hour))
		values_per_hour_array = values_per_hour.values.reshape((values_per_hour.size,1))

		sns.heatmap(values_per_hour_array[0:], xticklabels=False, yticklabels=False, cbar_kws={'label': 'Dose [mg]', 'format': '%d[mg]'}, cmap="YlGnBu", ax = axis)

		axis.set_ylabel("Hour of Day")
		plt.yticks(rotation=45)


		## Want to set tight layout:
		# top=0.957,
		# bottom=0.081,
		# left=0.104,
		# right=0.833,
		# hspace=0.2,
		# wspace=0.597
		# plt.tight_layout()
		plt.tight_layout()

		# Only export (save) the figure to file if the exportFormat isn't None
		if exportFormat:
			figure.savefig(figure_name, format=exportFormat, dpi=1200)



