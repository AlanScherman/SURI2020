#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np
import scipy.stats as sts
from matplotlib import pyplot as plt

def ripley_K(data_x, data_y, edge_x, edge_y, cross_data_x = [], cross_data_y = [], radii = [], mode = 'none', transform = None, cv = True):
    
    """Calculates and plots regular or cross Ripley K curve for subsurface data. 
       Plots K curve with confidence envelopes defined by P25 and P975. Includes options for L or H transform functions.
       Regular Ripley K calculates the behavior of events defined by data_x and data_y about themselves.
       Cross Ripley K calculates the behavior of events defined by data_x and data_y about events
       defined by cross_data_x and cross_data_y.
    
    :param data_x: events x-coordinates
    :type: pandas.DataFrame
    :param data_y: events y-coordinates
    :type: pandas.DataFrame
    :param edge_x: boundary point x-coordinates
    :type: pandas.DataFrame
    :param edge_y: boundary point y-coordinates
    :type: pandas.DataFrame
    :param cross_data_x: default empty array for regular K; 
                         for cross K, pandas.DataFrame w/ x-coordinates of events from which to calculate K
    :param cross_data_y: default empty array for regular K; 
                         for cross K, pandas.DataFrame w/ y-coordinates of events from which to calculate K
    :param radii: default empty array;
                  pandas.DataFrame for custom search radii
    :param mode: default 'none' for no edge correction; 
                 adjustments available: 'translation','ohser','ripley'
    :param transform: default None for plot of K curve; 
                      'L' for L-transform and 'H' for H-transform
    :param cv: default True used for recursive call; 
               if False then confidence interval is not calculated
    :return:K_plot: plot of Ripley K curve with confidence intervals
           :type: figure
    
    Note: Source code for regular K calculation and edge corrections taken from Astropy v4.0.1 class RipleysKEstimator,
          created by the Astropy Developers, last updated on 02 Apr 2020, available at: 
          "https://docs.astropy.org/en/stable/_modules/astropy/stats/spatial.html#RipleysKEstimator".
    """
    
    # Converting to NumPy row format
    
    data_x = np.asarray(data_x).reshape(len(data_x),1)
    data_y = np.asarray(data_y).reshape(len(data_y),1)
    
    cross_data_x = np.asarray(cross_data_x).reshape(len(cross_data_x),1)
    cross_data_y = np.asarray(cross_data_y).reshape(len(cross_data_y),1)
        
    radii = np.asarray(radii).reshape(len(radii),1)
        
    edge_x = np.asarray(edge_x).reshape(len(edge_x),1)
    edge_y = np.asarray(edge_y).reshape(len(edge_y),1)

    # Exception handling
    
    if cv == True:
        if cross_data_x.size>0 and cross_data_y.size>0:
            if np.shape(cross_data_x)[1] != 1:
                raise TypeError('"cross_data_x" array should be n-by-1 and represent cross events coordinates')
            if np.shape(cross_data_y)[1] != 1:
                raise TypeError('"cross_data_y" array should be n-by-1 and represent cross events coordinates')
            if len(cross_data_x) != len(cross_data_y):
                raise TypeError('"cross_data_x" and "cross_data_y" should have same dimensions')
        if np.shape(data_x)[1] != 1:
            raise TypeError('"data_x" array should be n-by-1 and represent events x-coordinates')
        if np.shape(data_y)[1] != 1:
            raise TypeError('"data_y" array should be n-by-1 and represent events y-coordinates')
        if len(data_x) != len(data_y):
            raise TypeError('"data_x" and "data_y" should have same dimensions')
        if np.shape(edge_x)[1] != 1:
            raise TypeError('"edge_x" array should be n-by-1 and represent boundary pts x-coordinates')
        if np.shape(edge_y)[1] != 1:
            raise TypeError('"edge_y" array should be n-by-1 and represent bpundary pts y-coordinates')
        if len(edge_x) != len(edge_y):
            raise TypeError('"edge_x" and "edge_y" should have same dimensions')
        if radii.size>0 and np.shape(radii)[1] != 1:
            raise TypeError('"radii" array should be n-by-1')
        if transform != None and transform != 'L' and transform != 'H':
            raise TypeError('Invalid transform selection')    
        
    # Determining limits and area of study
    
    edge = np.column_stack((edge_x,edge_y))
    
    x_min = min(edge[:,0]); x_max = max(edge[:,0])
    y_min = min(edge[:,1]); y_max = max(edge[:,1])
    
    area = 0.5*np.abs(np.dot(edge[:,0],np.roll(edge[:,1],1)) - np.dot(edge[:,1],np.roll(edge[:,0],1)))  # Implementation of shoelace formula

    # Compute default radii
    
    if cv == True and radii.size == 0:
        if abs(x_max - x_min) < abs(y_max - y_min):
            radii = (np.linspace(0,x_max-x_min,50)).reshape(50,1)
        else:
            radii = (np.linspace(0,y_max-y_min,50)).reshape(50,1)
        
    # Pairwise differences
    
    npts = len(data_x)
    data = np.column_stack((data_x,data_y))
    
    if cross_data_x.size>0: # Cross K
        cross_npts = len(cross_data_x)
        cross_data = np.column_stack((cross_data_x,cross_data_y))
        diff = np.zeros(shape=(npts*cross_npts,2), dtype=np.double)
        for i in range(cross_npts):             
            sub_data = np.zeros(shape=(npts,2))
            sub_data[:] = cross_data[i]
            diff[npts*i:npts*(i+1)] = abs(sub_data - data)
    
    
    else: # Regular K - (Version 4.0.1)[RipleysKEstimator].www.astropy.org
        diff = np.zeros(shape=(npts*(npts-1)//2,2), dtype=np.double)
        k = 0
        for i in range(npts - 1):
            size = npts - i - 1
            diff[k:k + size] = abs(data[i] - data[i+1:])
            k += size
            
    # Evaluation

    ripley = np.zeros(len(radii))
    dist = np.hypot(diff[:,0],diff[:,1])
    
    if mode == 'none':
        for r in range(len(radii)):
            ripley[r] = (dist<radii[r]).sum()

        if cross_data_x.size>0:    # Cross K
            ripley = area * ripley / (npts * cross_npts)
        else:                      # Regular K - (Version 4.0.1)[RipleysKEstimator].www.astropy.org
            ripley = area * 2. * ripley / (npts * (npts - 1))
       
    elif mode == 'translation': # Any area geometry
        intersec_area = (((x_max - x_min) - diff[:, 0]) * ((y_max - y_min) - diff[:, 1]))

        for r in range(len(radii)):
            dist_indicator = dist < radii[r]
            ripley[r] = ((1 / intersec_area) * dist_indicator).sum()
            
        if cross_data_x.size>0:    # Cross K
            ripley = (area**2 / (npts * cross_npts)) * ripley
        else:                      # Regular K - (Version 4.0.1)[RipleysKEstimator].www.astropy.org
            ripley = (area**2 / (npts * (npts - 1))) * 2 * ripley
        
       
    elif mode == 'ohser': # (Version 4.0.1)[RipleysKEstimator].www.astropy.org
        a = area
        b = max((y_max - y_min) / (x_max - x_min), (x_max - x_min) / (y_max - y_min))
        x = dist / np.sqrt(a / b)
        u = np.sqrt((x * x - 1) * (x > 1))
        v = np.sqrt((x * x - b ** 2) * (x < np.sqrt(b ** 2 + 1)) * (x > b))
        c1 = np.pi - 2 * x * (1 + 1 / b) + x * x / b
        c2 = 2 * np.arcsin((1 / x) * (x > 1)) - 1 / b - 2 * (x - u)
        c3 = (2 * np.arcsin(((b - u * v) / (x * x)) * (x > b) * (x < np.sqrt(b ** 2 + 1))) + 2 * u + 2 * v / b - b - (1 + x * x) / b)
        cov_func = ((a / np.pi) * (c1 * (x >= 0) * (x <= 1) + c2 * (x > 1) * (x <= b) + c3 * (b < x) * (x < np.sqrt(b ** 2 + 1))))
        
        for r in range(len(radii)):
            dist_indicator = dist < radii[r]
            ripley[r] = ((1 / cov_func) * dist_indicator).sum()

        if cross_data_x.size>0:    # Cross K
            ripley = (area**2 / (npts * cross_npts)) * ripley
        else:                      # Regular K
            ripley = (area**2 / (npts * (npts - 1))) * 2 * ripley

    elif mode == 'ripley': # Assumes polygonal area of study
          
        if cross_data_x.size>0:  # Cross K
            hor_dist = np.zeros(shape=(npts*cross_npts), dtype=np.double)
            ver_dist = np.zeros(shape=(npts*cross_npts), dtype=np.double)
            
            for k in range(cross_npts - 1): 
                min_hor_dist = min(x_max - cross_data[k][0], cross_data[k][0] - x_min)
                min_ver_dist = min(y_max - cross_data[k][1], cross_data[k][1] - y_min)
                start = k*npts
                end = (k+1)*npts
                hor_dist[start: end] = min_hor_dist * np.ones(npts)
                ver_dist[start: end] = min_ver_dist * np.ones(npts)
                
        else: # Regular K - (Version 4.0.1)[RipleysKEstimator].www.astropy.org
            hor_dist = np.zeros(shape=(npts * (npts - 1)) // 2, dtype=np.double)
            ver_dist = np.zeros(shape=(npts * (npts - 1)) // 2, dtype=np.double)

            for k in range(npts - 1):
                min_hor_dist = min(x_max - data[k][0], data[k][0] - x_min)
                min_ver_dist = min(y_max - data[k][1], data[k][1] - y_min)
                start = (k * (2 * (npts - 1) - (k - 1))) // 2
                end = ((k + 1) * (2 * (npts - 1) - k)) // 2
                hor_dist[start: end] = min_hor_dist * np.ones(npts - 1 - k)
                ver_dist[start: end] = min_ver_dist * np.ones(npts - 1 - k)
            
        dist_ind = dist <= np.hypot(hor_dist, ver_dist)

        w1 = (1 - (np.arccos(np.minimum(ver_dist, dist) / dist) + np.arccos(np.minimum(hor_dist, dist) / dist)) / np.pi)
        w2 = (3 / 4 - 0.5 * (np.arccos(ver_dist / dist * ~dist_ind) + np.arccos(hor_dist / dist * ~dist_ind)) / np.pi)

        weight = dist_ind * w1 + ~dist_ind * w2

        for r in range(len(radii)):
            ripley[r] = ((dist < radii[r]) / weight).sum()

        if cross_data_x.size > 0:
            ripley = area * ripley / (npts * cross_npts)
        else:
            ripley = area * 2. * ripley / (npts * (npts - 1))
    

    if cv == True: # Recursive parameter
        
        # Confidence envelopes

        xDel = x_max - x_min
        yDel = y_max - y_min

        Ks = np.zeros((len(radii),1000))

        if cross_data_x.size>0: # Cross K
            
            for i in range(1000): 
                x_coords = xDel*sts.uniform.rvs(0,1,((npts,1))) + x_min 
                y_coords = yDel*sts.uniform.rvs(0,1,((npts,1))) + y_min 
                cross_x_coords = xDel*sts.uniform.rvs(0,1,((cross_npts,1))) + x_min 
                cross_y_coords = yDel*sts.uniform.rvs(0,1,((cross_npts,1))) + y_min 
                
                Ks[:,i] = ripley_K(data_x=x_coords, data_y=y_coords, edge_x=edge_x, edge_y=edge_y,                                   cross_data_x=cross_x_coords, cross_data_y=cross_y_coords, radii=radii,                                   mode=mode, transform=transform, cv=False) # Recursive call
                
            k25 = np.zeros((len(radii),1))
            k975 = np.zeros((len(radii),1))
            
            for j in range(len(radii)):
                Ks[j,:] = np.sort(Ks[j,:])
                k25[j] = Ks[j,:][24]
                k975[j] = Ks[j,:][974]

        else: # Regular K

            for i in range(1000): 
                x_coords = xDel*sts.uniform.rvs(0,1,((npts,1))) + x_min
                y_coords = yDel*sts.uniform.rvs(0,1,((npts,1))) + y_min

                Ks[:,i] = ripley_K(data_x=x_coords, data_y=y_coords, edge_x=edge_x, edge_y=edge_y,                                   radii=radii, mode=mode, transform=transform, cv=False)      # Recursive call

            k25 = np.zeros((len(radii),1))
            k975 = np.zeros((len(radii),1))

            for j in range(len(radii)):
                Ks[j,:] = np.sort(Ks[j,:])
                k25[j] = Ks[j,:][24]
                k975[j] = Ks[j,:][974]

        # Transforms
    
        if transform == 'L':
            ripley = np.sqrt(ripley/np.pi)
            k25 = np.sqrt(k25/np.pi)
            k975 = np.sqrt(k975/np.pi)
        if transform == 'H':
            ripley = np.sqrt(ripley/np.pi) - radii[:,0]
            k25 = np.sqrt(k25/np.pi) - radii
            k975 = np.sqrt(k975/np.pi) - radii   
               
        # Plotting 
    
        K_plot = plt.figure(figsize=(16,12))
        plt.style.use('fivethirtyeight')
        plt.xlabel('Radii')
        
        if cross_data_x.size>0:
            if transform == None: plt.title('Cross Ripley K \n Edge correction: "{ec}"'.format(ec=mode)), plt.ylabel('K Value')
            if transform == 'L': plt.title('L Transform of Cross Ripley K \n Edge correction: "{ec}"'.format(ec=mode)), plt.ylabel('L Value') 
            if transform == 'H': plt.title('H Transform of Cross Ripley K \n Edge correction: "{ec}"'.format(ec=mode)), plt.ylabel('H Value')
        else:
            if transform == None: plt.title('Regular Ripley K \n Edge correction: "{ec}"'.format(ec=mode)), plt.ylabel('K Value')
            if transform == 'L': plt.title('L Transform of Ripley K \n Edge correction: "{ec}"'.format(ec=mode)), plt.ylabel('L Value') 
            if transform == 'H': plt.title('H Transform of Ripley K \n Edge correction: "{ec}"'.format(ec=mode)), plt.ylabel('H Value')
        
        plt.plot(radii, ripley, color = 'black', label='K', linewidth=2)
        plt.plot(radii, k25, color = 'red', label = 'P25', linewidth=2)
        plt.plot(radii, k975, color = 'blue', label = 'P975', linewidth=2)
        plt.fill_between(radii[:,0], k25[:,0], k975[:,0], color = 'yellow', alpha = 0.5)
        
        plt.legend()
        plt.show()
        
    else:
        return ripley

