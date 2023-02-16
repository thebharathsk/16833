'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from multiprocessing import Pool, Process, Queue
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit   = 1.
        self._z_short = 0.1
        self._z_max   = 0.1
        self._z_rand  = 100.

        self._sigma_hit = 50.
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting in degrees
        self._subsampling = 1
        
        # Number of processes to be used for subsampling
        self.num_processes = 8
        
        # Store oocupancy map
        self.map = occupancy_map
        
        # Dimensions of occupancy map
        self.h, self.w = self.map._occupancy_map.shape
        
        # Distance that ray can skip during casting iterations
        self.ray_skip_dist = 10
        
        #separation between laser and robot center in cm
        self.laser_loc = 25

        # Store rays
        self.rays = self.map._occupancy_map.copy()
        
    def sensor_location(self, x):
        """Find sensor's position based on state of robot

        Args:
            x (list): state of robot represented by a particle
        """
        #get x, y, theta of robot from state
        rx_cm, ry_cm, theta_rad = x
        
        #compute laser's location
        lx_cm = rx_cm + self.laser_loc*np.cos(theta_rad)
        ly_cm = ry_cm + self.laser_loc*np.sin(theta_rad)
        
        return [lx_cm, ly_cm, theta_rad]
    
    def ray_casting(self, x, angle_rad):
        """ray casting algorithm to find true range 

        Args:
            x (list): state of the range sensor represented by a particle
            angle (float): angle of laser beam
        """
        if angle_rad > np.pi or angle_rad <-np.pi:
            raise ValueError("Angle must be in radians and in the range [-pi, pi]")

        #unpack particle
        lx, ly, ltheta = x
        
        #measure angle of laser beam in global reference frame
        theta = (ltheta + angle) - math.pi/2
        
        #perform ray casting
        #initialize positions to positions of range sensor
        cx, cy = lx, ly
        
        #iterate till wall is hit or the ray reaches edge of map
        while True:
            #convert cm into px
            cx_p = math.floor(cx/self.map._resolution)
            cy_p = math.floor(cy/self.map._resolution)
            
            #if ray goes beyond edge
            if cx_p < 0 or cx_p > self.w - 1 or \
                cy_p < 0 or cy_p > self.h - 1:
                    break
            
            #if ray hits the wall
            if self.map._occupancy_map[cy_p][cx_p] != 0:
                break
            
            #update x and y coordinates
            cx = cx + self.ray_skip_dist*np.cos(theta)
            cy = cy + self.ray_skip_dist*np.sin(theta)
            
            self.rays[cy_p][cx_p] = -2
            #print("ray = ", cy, cx, self.rays[cy_p][cx_p])
            
        z_gt = math.sqrt((cx-lx)**2 + (cy-ly)**2)

        return z_gt_cm
    
    def get_true_ranges(self, x):
        """function to find true range for a given state(x,y,theta) of robot

        Args:
            x (list): state of range sensor
        """
        #array to store true ranges
        z_gt_cm = []
        
        #iterate based on subsampling
        for i in range(0, 180, self._subsampling):
            #compute range using ray casting
            z_gt_angle = self.ray_casting(x, i*(math.pi/180))
                        
            #add range to list
            z_gt.append(z_gt_angle)
        
        return z_gt
    
    def get_true_ranges_vectorized(self, x):
        """function to find true range for a given state(x,y,theta) of robot using vectorized implementation

        Args:
            x (list): state of range sensor
        """
        #array to store true ranges
        z_gt = []
        
        #store angles at which rays are cast
        angles = np.arange(1, 181, self._subsampling)
        
        #create list of arguments
        args = Queue()
        results = Queue()
        for a in angles:
            args.put((x, a*(math.pi/180)))
        
        #add
        chunk_size = len(arg_list)//self.num_processes
        arg_list_split = []
        arg_list_split = [arg_list[x:x+chunk_size] for x in range(0, len(arg_list), chunk_size)]
        
        #paralelized ray casting
        processes = []
        #start processes
        for args in arg_list_split:
            p = Process(target = self.ray_casting, args=(args,))
            processes.append(p)
            p.daemon = True
            p.start()

        #end processes
        for p in processes:
            p.join()
        
        p = Pool(self.num_processes)
        z_gt = p.starmap(self.ray_casting, arg_list)       
        
        return z_gt
    
    def calc_p_hit(self, z_t, z_gt):
        """
        Probability of measuring the true range of an object
        Model this as a gaussian centered around the ray cast and has spread via a hyper-param
        """
        p_hit = 0.
        norm_cdf = norm.cdf(self._max_range, loc = z_gt, scale = self._sigma_hit)
        if 0 <= z_t and z_t <= self._max_range and norm_cdf > 0:
            eta = 1 / norm_cdf
            p_hit = eta * math.exp(-0.5 * ((z_t - z_gt) / self._sigma_hit) ** 2)
        return p_hit

    def calc_p_short(self, z_t, z_gt):
        """
        Calculate the probability of unexpected objects.
        We will treat these objects as sensor noise, they will most commonly be close to the sensor
        """
        p_short = 0.
        if z_t <= z_gt:
            eta = 1 / (1 - math.exp(-self._lambda_short * z_gt))
            p_short = eta * self._lambda_short * math.exp(-self._lambda_short * z_t)
        return p_short

    def calc_p_max(self, z_t):
        """
        Calculate the probability of a sensor failure AKA max-range measurement.
        Count the number of maximum measurements
        """
        p_rand = 0.
        if z_t < self._max_range:
            p_rand = float(z_t == self._max_range)
        return p_rand

    def calc_p_rand(self):
        """
        Calculate the probability of random measurements.
        This could be anything from phantom readings to reflectance to cross talk
        """
        return 1/self._max_range

    def sensor_probs(self, z_t, z_gt):
        """function to compute probabiltiies of measurement

        Args:
            z_t (list): measured data
            z_gt (list): ground truth data
        """
        #initialize probabilities
        p1, p2, p3, p4 = [],[],[],[]
        
        #HIT PROBABILITY
        #initialize probabilties
        p1 = np.zeros(len(z_t))
        
        #mask for non zero probabilities
        mask = z_t <= self._max_range
        
        #compute normalization factors
        n = 1/norm.cdf(self._max_range, loc=z_gt[mask], scale=self._sigma_hit)
        #         #compute probability
        #         prob = n*math.exp(-0.5*((p-q)/self._sigma_hit)**2)
            
        #compute probabilities
        p1[mask] = n*np.exp(-0.5*((z_t[mask]-z_gt[mask])/self._sigma_hit)**2)
                        
        #SHORT PROBABILITY
        #initialize probabilties    
        p2 = np.zeros(len(z_t))
        #     if p <= q:
        #         n = 1/(1 - math.exp(-self._lambda_short*q))
        #         prob =  n*self._lambda_short*math.exp(-self._lambda_short*p)
        #     p2.append(prob)
        
        #mask for non zero probabilities
        mask = z_t <= z_gt

        #compute normalization factors
        n = 1/(1 - math.exp(-self._lambda_short*z_gt[mask]))
        
        #compute probabilties
        p2[mask] =  n*self._lambda_short*np.exp(-self._lambda_short*z_t[mask])
            
        
        #MAX PROBABILITY
        #initialize probabilties    
        p3 = np.zeros(len(z_t))
        
        #mask for non zero probabilities
        mask = z_t > self._max_range
        
        #compute probabilities
        p3[mask] = 1
        
        #RAND PROBABILITY
        #initialize probabilties    
        p4 = np.zeros(len(z_t))
        
        #mask for non zero probabilities
        mask = z_t < self._max_range
        
        #compute probabilities
        p4[mask] = 1/(self._max_range)
                
        # #rand probability
        # for (p, q) in zip(z_t, z_gt):
        #     prob = 0
        #     if p < self._max_range:
        #         prob = 1/(self._max_range)
                
        #     p4.append(prob)
        
        return np.array(p1), np.array(p2), np.array(p3), np.array(p4)
        

    def beam_range_finder_model(self, z_t1_cm, x_t1):
        """
        param[in] z_t1_cm : laser range readings [array of 180 values] at time t in centimeters
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        prob_zt1 = 1.0
        
        #find laser position based on robot position
        x_sensor = self.sensor_location(x_t1)
        
        #perform ray casting to find GT ranges at various angles
        z_gt_cm = self.get_true_ranges(x_sensor)
        
        #sample laser measurements
        z_t1_cm = z_t1_cm[::self._subsampling]
        
        #find probabilities
        p1, p2, p3, p4 = self.sensor_probs(z_t1_cm, z_gt_cm)
        
        # TODO: logsumexp for numerical stability
        #aggregate probabilities
        # Validate the sum of the mixing parameters is 1
        sum_z = self._z_hit + self._z_short + self._z_max + self._z_rand

        p = (self._z_hit   / sum_z) * p1 + \
            (self._z_short / sum_z) * p2 + \
            (self._z_max   / sum_z) * p3 + \
            (self._z_rand  / sum_z) * p4
        
        #sum of log probabilities
        prob_log = np.sum(np.log(p))
        
        return prob_log


if __name__ == "__main__":
    src_path_map = 'C:/Users/chris/dev/16833/hw1/src/data/map/wean.dat'
    map1 = MapReader(src_path_map)

    sm = SensorModel(map1)
    
    t1 = time.time()
    xx = 4110
    yy = 5130
    for n in range(500):
        z_gt_1 = sm.get_true_ranges([xx, yy, math.pi/2])
        t2 = time.time()
        
    sm.rays[yy//10, xx//10] = -5
    # t3 = time.time()
    # z_gt_2 = sm.get_true_ranges_vectorized([590, 145, 0])
    # t4 = time.time()
    
    print(z_gt_1[0::10])
    print(len(z_gt_1))
    # print(z_gt_2[0::10])
    
    print(t2-t1)
    # print(t4-t3)
    
    # Visualize the map with the rays
    plt.imshow(sm.rays)
    map1.visualize_rays(sm.rays)
    plt.savefig('./rays.png')