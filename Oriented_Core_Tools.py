# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:00:29 2019
Last modification on Wed Sep 25 13:00:29 2019

@author: acate
"""

import numpy as np
import math
from scipy.linalg import expm, norm

"""
Series of functions derived from the android app for core measurement.
"""

def polarToCartesian(plunge, trend):
    plunge = np.radians(plunge)
    trend = np.radians(trend)
    x = np.cos(plunge) * np.sin(trend)
    y = np.cos(plunge) * np.cos(trend)
    z = -np.sin(plunge)
    return np.array([x, y, z])


def cartesianToPolar(vector):
    plunge = -np.degrees(np.arcsin(vector[2]))
    trend = (-np.degrees(np.arctan2(vector[1], vector[0]))+90)%360
    if plunge<0:
        plunge = -plunge
        trend = (trend+180)%360
    return plunge, trend

"""
Function form https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
Returns the rotation matrix given an axis and a theta angle IN RADIANS
Only need to apply np.dot(Matrix, Vector) to apply the rotation to a vector
"""
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))

"""
Functions to estimate dip and dip direction from alpha, beta and gamma and core orientation.
The functions are not vectorized yet and take only a single input point.
Need to vectorize the functions to use numpy arrays as inputs.
"""
# Alpha Beta to Dip Dip Direction
def ABtoDDD(corePlunge, coreTrend, alpha, beta):
    # get core axis vector
    coreAxis = polarToCartesian(corePlunge, coreTrend)
    # get horizontal perp to core axis for second rotation
    corePerp = polarToCartesian(0, (coreTrend+90)%360)
    # get vector with alpha = 0 and beta = 0
    v = polarToCartesian(90-corePlunge, (coreTrend+180)%360)
    # get alpha and beta of pole to plane
    alphaPole, betaPole = 90-alpha, (beta+180)%360
    # rotatation 1
    axis1, theta1 = coreAxis, math.radians(betaPole)
    M1 = M(axis1, theta1)
    pole1 = np.dot(M1, v)
    corePerp1 = np.dot(M1, corePerp)
    # rotation 2
    axis2, theta2 = corePerp1, math.radians(90-alphaPole)
    M2 = M(axis2, theta2)
    pole2 = np.dot(M2, pole1)
    # get pole to plane
    polePlunge, poleTrend = cartesianToPolar(pole2)
    dip = 90-polePlunge
    dipDirection = (poleTrend+180)%360
    return dip, dipDirection

# alpha beta to plunge and bearing
def ABtoPB(corePlunge, coreTrend, alpha, beta):
    # get core axis vector
    coreAxis = polarToCartesian(corePlunge, coreTrend)
    # get second perp to core axis for second rotation
    corePerp = polarToCartesian(0, (coreTrend+90)%360)
    # get vector with alpha = 0 and beta = 0
    v = polarToCartesian(90-corePlunge, (coreTrend+180)%360)
    # rotatation 1
    axis1, theta1 = coreAxis, math.radians(beta)
    M1 = M(axis1, theta1)
    line1 = np.dot(M1, v)
    corePerp1 = np.dot(M1, corePerp)
    # rotation 2
    axis2, theta2 = corePerp1, math.radians(90-alpha)
    M2 = M(axis2, theta2)
    line2 = np.dot(M2, line1)
    # get line orientation
    plunge, bearing = cartesianToPolar(line2)
    return plunge, bearing

# alpha beta gamma to plunge and bearing
def ABGtoPB(corePlunge, coreTrend, alpha, beta, gamma):
    # get core axis vector
    coreAxis = polarToCartesian(corePlunge, coreTrend)
    # get perp to core axis for second rotation
    corePerp = polarToCartesian(0, (coreTrend+90)%360)
    # get vector with alpha = 0 and beta = 0
    v = polarToCartesian(90-corePlunge, (coreTrend+180)%360)
    # get alpha and beta of pole to plane
    alphaPole, betaPole = 90-alpha, (beta+180)%360
    
    # rotatation 1
    axis1, theta1 = coreAxis, math.radians(betaPole)
    M1 = M(axis1, theta1)
    pole1 = np.dot(M1, v)
    corePerp1 = np.dot(M1, corePerp)
    # rotation 2
    axis2, theta2 = corePerp1, math.radians(90-alphaPole)
    M2 = M(axis2, theta2)
    pole2 = np.dot(M2, pole1)
    
    # get alpha and beta of line gamma = 0 (Long axis line)
    alphaLong, betaLong = alpha, beta
    # rotation 1 of long axis
    axis3, theta3 = coreAxis, math.radians(betaLong)
    M3 = M(axis3, theta3)
    line3 = np.dot(M3, v)
    corePerp3 = np.dot(M3, corePerp)
    # rotation 2 of long axis
    axis4, theta4 = corePerp3, math.radians(90-alphaLong)
    M4 = M(axis4, theta4)
    line4 = np.dot(M4, line3)
    # rotation of long axis around the pole of gamma
    axis5, theta5 = pole2, math.radians(gamma)
    M5 = M(axis5, theta5)
    line5 = np.dot(M5, line4)
    
    # get line orientation    
    plunge, bearing = cartesianToPolar(line5)
    return plunge, bearing

"""
Function to get core orientation (plunge and trend) at each measurement as numpy arrays
Inputs are corresponding columns from the survey and structural measurement tables
As a convention, positive plunges are downward holes
Attributes the closest survey value to each point. Improvement would estimate core orietnation at a given depth
using a linear combination of the survey points above and below.
"""
def getCoreOrientation(surveyHole, surveyDepth, surveyPlunge, surveyTrend, pointHole, pointDepth):
    surveyHole = np.array(surveyHole)
    surveyDepth = np.array(surveyDepth)
    surveyPlunge = np.array(surveyPlunge)
    surveyTrend = np.array(surveyTrend)
    pointHole = np.array(pointHole)
    pointDepth = np.array(pointDepth)
    plunges = []
    trends = []
    for i in range(len(pointDepth)):
        hole = pointHole[i]
        depth = pointDepth[i]
        holeDepths = surveyDepth[surveyHole==hole]
        # within the survey of the hole, takes the closest survey point to a depth and extracts its plunge
        holePlunges = surveyPlunge[surveyHole==hole]
        plunges.append(holePlunges[np.argmin(np.abs(np.array(holeDepths)-depth))])
        # within the survey of the hole, takes the closest survey point to a depth and extracts its trend
        holeTrends = surveyTrend[surveyHole==hole]
        trends.append(holeTrends[np.argmin(np.abs(np.array(holeDepths)-depth))])
    return np.array(plunges), np.array(trends)

"""
Other functions
"""

def DDtoStrike(dipDir):
    strike = (dipDir-90)%360
    return strike

def StriketoDD(strike):
    dipDir = (strike+90)%360
    return dipDir