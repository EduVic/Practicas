# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:44:28 2018

@author: acate
"""
import matplotlib.pyplot as plt
import mplstereonet
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from scipy.stats import circstd


def f(angle):
    angle = (angle - 90) % 360
    return angle
dip_dir_to_strike = np.vectorize(f)

class Downhole_Structural_Data ():
    def __init__(self, df=None, df_Hole_ID=None, df_Depth=None, df_Dip=None, df_DipDir=None,
                        df_Type=None, df_Alpha=None, df_Beta=None,
                        collar=None, collar_Hole_ID=None, survey=None,
                         survey_Hole_ID=None, survey_Depth=None,
                          survey_Dip=None, survey_DipDir=None):
        self.df = df
        self.df_Hole_ID = df_Hole_ID
        self.df_Depth = df_Depth
        self.df_Dip = df_Dip
        self.df_DipDir = df_DipDir
        self.df_Type = df_Type
        self.df_Alpha = df_Alpha
        self.df_Beta = df_Beta
        self.collar = collar
        self.collar_Hole_ID = collar_Hole_ID
        self.survey = survey
        self.survey_Hole_ID = survey_Hole_ID
        self.survey_Depth = survey_Depth
        self.survey_Dip = survey_Dip
        self.survey_DipDir = survey_DipDir
    def plot_holes_stereo(self, category=None, save=False):
        if category is not None:
            temp_df = self.df[self.df[self.df_Type]==category]
        else:
            temp_df = pd.DataFrame(self.df)
        for hole, group in temp_df.groupby(self.df_Hole_ID):
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111, projection='stereonet')
            if category is not None:
                ax.set_title('%s in %s' % (category, hole), fontsize=16, y=1.07)
            else:
                ax.set_title('All data in %s' % (hole), fontsize=16, y=1.07)

            density = ax.density_contourf(dip_dir_to_strike(group[self.df_DipDir]), group[self.df_Dip], method='schmidt',
                                          measurement='poles', cmap='Oranges')

            # Plot varying orientations of the hole due to deviation
            survey_points = ax.line(plunge=-self.survey[self.survey[self.survey_Hole_ID]==hole][self.survey_Dip],
                                    bearing=self.survey[self.survey[self.survey_Hole_ID]==hole][self.survey_DipDir],
                                    marker='.', c='blue', label='Survey')

            # Plot hole collar orientation
            DIP = -self.survey[self.survey_Dip][self.survey[self.survey_Hole_ID]==hole][0]
            AZ = self.survey[self.survey_DipDir][self.survey[self.survey_Hole_ID]==hole][0]
            ax.line(DIP, AZ, marker='o', color='blue', markeredgecolor='black', markersize=10, label='Collar')

            # Plot cone based on median alpha angle
            angle = 90 - np.median(group[self.df_Alpha])
            ax.cone(DIP, AZ, angle, facecolor='', linewidth=2, edgecolors='red')

            # Plot color bar
            cbaxes = fig.add_axes([1, 0.1, 0.03, 0.8])
            cbar = fig.colorbar(density, cax=cbaxes)
            cbar.set_label('Point density (exponential_kamb)')
            cbaxes.text(7, 0.5,'Azi: %d - Dip: %d - Equal Area Lower Hemisphere / n=%d' % (AZ, DIP, len(group)))
            if (save == True):
                if category is not None:
                    filename = hole + '_density_' + category + '.png'
                else :
                    filename = hole + '_density_all.png'
                plt.savefig(filename, format='png', bbox_inches='tight', dpi=400)
            plt.show()

    def plot_holes_down(self, category=None, save=False):
        print('HAVE TO REDO IT BASED ON CODE IN THE MAKE REPORT METHOD')
#         if category is not None:
#             temp_df = self.df[self.df[df_Type]==category]
#         else:
#             temp_df = pd.DataFrame(self.df)
#         for hole, group in temp_df.groupby(df_Hole_ID):
#             fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(3,7))
#             ax[0].plot(group[self.df_Alpha], group[self.df_Depth], color='orange', linewidth=1, marker='.', markersize=5)
#             ax[0].set(xlabel='Alpha')
#             start, end = ax[0].get_ylim()
#             start = start - (start%10)
#             end = end + (end%10)
#             stepsize=10
#             ax[0].yaxis.set_ticks(np.arange(start, end, stepsize))
#             ax[0].xaxis.set_ticks([0, 45, 90])
#             ax[1].plot(group[self.df_Beta], group[self.df_Depth], color='orange', linewidth=1, marker='.', markersize=5)
#             ax[1].set(xlabel='Beta')
#             ax[1].xaxis.set_ticks([0, 180, 360])
#             fig.suptitle(hole, fontsize=12, y=0.95)
#             plt.gca().invert_yaxis()
#             plt.show()

    def plot_holes_stereo_depth(self, category=None, save=False):
        if category is not None:
            temp_df = self.df[self.df[self.df_Type]==category]
        else:
            temp_df = pd.DataFrame(self.df)
        for hole, group in temp_df.groupby(self.df_Hole_ID):
            # create figure
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111, projection='stereonet')
            ax.set_title('log_core_orientation - %s ' % (hole), fontsize=16, y=1.1)
            # plot structural measurements
            lon, lat = mplstereonet.stereonet_math.pole(dip_dir_to_strike(group[self.df_DipDir]), group[self.df_Dip])
            points = ax.scatter(lon, lat, c=group[self.df_Depth], marker='o', s=20, cmap='jet', edgecolor='black')
            # plot hole collar
            DIP = -self.survey[self.survey_Dip][self.survey[self.survey_Hole_ID]==hole][0]
            AZ = self.survey[self.survey_DipDir][self.survey[self.survey_Hole_ID]==hole][0]
            ax.line(DIP, AZ, marker='o', markersize=10, color='black')
            # plot small circle
            angle = 90 - np.median(group[self.df_Alpha])
            ax.cone(DIP, AZ, angle, facecolor='', linewidth=2, edgecolors='red')
            # plot color bar
            cbaxes = fig.add_axes([1, 0.1, 0.03, 0.8])
            cbar = fig.colorbar(points, cax=cbaxes)
            cbar.set_label('depth along hole(m)')
            cbar.ax.invert_yaxis()
            # save
            if (save == True):
                if category is not None:
                    filename = hole + '_' + category + '.png'
                else:
                    filename = hole + '_AllStructures.png'
                plt.savefig(filename, format='png', bbox_inches='tight')
            # show graph
            plt.show()

    def create_report(self, category=None, save=False):
        if category is not None:
            temp_df = self.df[self.df[self.df_Type]==category]
        else:
            temp_df = pd.DataFrame(self.df)
        for hole, group in temp_df.groupby(self.df_Hole_ID):
            DIP = self.survey[self.survey_Dip][self.survey[self.survey_Hole_ID]==hole].iloc[0]
            AZ = self.survey[self.survey_DipDir][self.survey[self.survey_Hole_ID]==hole].iloc[0]
            angle = 90 - np.median(group[self.df_Alpha])
            ### Create empty diagram
            fig = plt.figure(figsize=(10,10))
            gs0 = gridspec.GridSpec(2, 1, height_ratios=[6,1])
            gs00 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0], width_ratios=[8,16,1])
            gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[1])
            gs000 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs00[0])
            gs001 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs00[1])
            gs002 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs00[2])

            ax00a = plt.subplot(gs000[0])
            ax00b = plt.subplot(gs000[1])
            ax010a = plt.subplot(gs001[0], projection='stereonet')
            ax010b = plt.subplot(gs002[0]) #cbaxes = fig.add_axes([1, 0.1, 0.03, 0.8])
            ax011a = plt.subplot(gs001[1], projection='stereonet')
            ax011b = plt.subplot(gs002[1]) #cbaxes = fig.add_axes([1, 0.1, 0.03, 0.8])
            ax012a = plt.subplot(gs01[0])

            ### Plot title
            ### Create df without missing Data
            groupna1 = group.dropna(subset=[self.df_Alpha, self.df_Depth,
                                    self.df_Beta])

            ### Plot downhole diagram
            ax00a.plot(groupna1[self.df_Alpha], groupna1[self.df_Depth], color='orange', linewidth=0, marker='.', markersize=5)
            ax00a.set(xlabel='Alpha')
            start = groupna1[self.df_Depth].min()
            start = start - (start%10)
            end = groupna1[self.df_Depth].max()
            end = end + 10 -(end%10)
            stepsize=10
            ax00a.yaxis.set_ticks(np.arange(start, end+10, stepsize))
            ax00a.xaxis.set_ticks([0, 45, 90])
            ax00a.invert_yaxis()
            ax00b.plot(groupna1[self.df_Beta], groupna1[self.df_Depth], color='orange', linewidth=0, marker='.', markersize=5)
            ax00b.yaxis.set_ticks(np.arange(start, end+10, stepsize))
            ax00b.set_yticklabels([])
            ax00b.set(xlabel='Beta')
            ax00b.xaxis.set_ticks([0, 180, 360])
            ax00b.invert_yaxis()
            ax00b.tick_params(axis='y', which='both', left='off', labelleft='off')

            ### Create df without missing Data
            groupna2 = group.dropna(subset=[self.df_Depth, self.df_DipDir, self.df_Dip])
            ### Plot stereonet density
            ax010a.set_title('Density countours', fontsize=12, y=1)
            # Plot density contours using schmidt
            density = ax010a.density_contourf(dip_dir_to_strike(groupna2[self.df_DipDir]), groupna2[self.df_Dip], method='schmidt',
                                          measurement='poles', cmap='Oranges')
            # Plot varying orientations of the hole due to deviation
            survey_points = ax010a.line(plunge=-self.survey[self.survey[self.survey_Hole_ID]==hole][self.survey_Dip],
                                    bearing=self.survey[self.survey[self.survey_Hole_ID]==hole][self.survey_DipDir],
                                    marker='.', c='blue', label='Survey')
            # Plot hole collar orientation
            ax010a.line(DIP, AZ, marker='o', color='blue', markeredgecolor='black', markersize=10, label='Collar')
            # Plot cone based on median alpha angle
            ax010a.cone(DIP, AZ, angle, facecolor='', linewidth=2, edgecolors='red')
            # Plot color bar
            cbar = plt.colorbar(density, cax=ax010b)
            cbar.set_label('Point density (exponential_kamb)')

            ### Plot stereonet depths
            ax011a.set_title('Points depth', fontsize=12, y=1)
            # plot structural measurements
            lon, lat = mplstereonet.stereonet_math.pole(dip_dir_to_strike(groupna2[self.df_DipDir]), groupna2[self.df_Dip])
            points = ax011a.scatter(lon, lat, c=groupna2[self.df_Depth], marker='o', s=20, cmap='jet', edgecolor='black')
            # plot hole collar
            ax011a.line(DIP, AZ, marker='o', markersize=10, color='black')
            # plot small circle
            ax011a.cone(DIP, AZ, angle, facecolor='', linewidth=2, edgecolors='red')
            # plot color bar
            cbar = plt.colorbar(points, cax=ax011b)
            ax011b.invert_yaxis()
            cbar.set_label('depth along hole(m)')

            ### Plot text
            # Title
            if category is not None:
                ax012a.text(0.01, 0.8, 'Quality report for hole %s and structures %s' % (hole, category), fontsize=12)
            else:
                ax012a.text(0.01, 0.8, 'Quality report for hole %s and all measurements' % (hole), fontsize=12)
            # plot stereonet charac
            ax012a.text(0.01, 0.55, 'Equal Area Lower Hemisphere')
            # plot hole orientation
            ax012a.text(0.01, 0.4, 'Collar orientation dip: %0.1f - azimuth: %0.1f' % (DIP, AZ))
            # plot number of samples
            ax012a.text(0.01, 0.25, 'n samples = %d' % (len(group)))
            # plot standard deviation
            Std_Dip = circstd(samples = group[self.df_Dip], low=0, high=360)
            Std_DipDir = circstd(samples = group[self.df_DipDir], low=0, high=360)
            Std_Alpha = circstd(samples = group[self.df_Alpha], low=0, high=90)
            Std_Beta = circstd(samples = group[self.df_Beta], low=0, high=360)
            ax012a.text(0.01, 0.1, 'Standard deviations: Dip: %0.2f - Dip Direction: %0.2f - Alpha: %0.2f - Beta: %0.2f'
                        % (Std_Dip, Std_DipDir, Std_Alpha, Std_Beta))
            ax012a.tick_params(axis='both', which='both', left='off', labelleft='off', bottom='off', labelbottom='off')

            ### Save and plot
            if (save == True):
                if category is not None:
                    filename = 'Quality report for hole ' + hole + ' and structures ' + category + '.png'
                else :
                    filename = 'Quality report for hole ' + hole + ' with all structures.png'
                plt.savefig(filename, format='png', bbox_inches='tight', dpi=400)
            plt.show()
