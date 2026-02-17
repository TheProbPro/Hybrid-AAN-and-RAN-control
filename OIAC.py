import numpy as np
import numpy.linalg as la

###########################################################################################
#                       Online Adaptive Impedance Control (OIAC) class                    #
# This class implements the OIAC controller. This code comes from the paper:              #
# "An Online Impedance Adaptation Controller for Decoding Skill Intelligence.             #
# Biomimetic Intelligence and Robotics, 3(2), [100100].                                   #
# https://doi.org/10.1016/j.birob.2023.100100",                                           #
#                                                                                         #
# and is copied from the following GitHub repository:                                     #
# "https://github.com/lenonrobot/Online-Impedance-Adaptation-Control/blob/main/oiac.py".  #
# The only changes from the original code is the values self.a, self.b, and self.k,       #
# which are the parameters of the OIAC controller.                                        #
###########################################################################################



class ada_imp_con():
    def __init__(self, dof):
        self.DOF = dof # degree of freedom

        self.k_mat = np.asmatrix(np.zeros((self.DOF, self.DOF))) # stiffness matrix
        self.b_mat = np.asmatrix(np.zeros((self.DOF, self.DOF))) # damping matrix

        self.ff_tau_mat = np.asmatrix(np.zeros((self.DOF, 1))) # feedforward torque matrix
        self.fb_tau_mat = np.asmatrix(np.zeros((self.DOF, 1))) # feedback torque matrix

        self.q = np.asmatrix(np.zeros((self.DOF, 1))) # position matrix
        self.q_d = np.asmatrix(np.zeros((self.DOF, 1))) # velocity matrix
        self.dq = np.asmatrix(np.zeros((self.DOF, 1))) # acceleration matrix
        self.dq_d = np.asmatrix(np.zeros((self.DOF, 1))) # desired velocity matrix

        self.a = 1.0
        self.b = 0.1
        self.k = 0.005
        
    def update_impedance(self, q, q_d, dq, dq_d):
        # copy inputs
        self.q = np.asmatrix(np.copy(q))
        self.q_d = np.asmatrix(np.copy(q_d))
        self.dq = np.asmatrix(np.copy(dq))
        self.dq_d = np.asmatrix(np.copy(dq_d))
        #Update matrices
        self.k_mat = (self.gen_track_error() * self.gen_pos_error().T)/self.gen_ad_factor()
        self.b_mat = (self.gen_track_error() * self.gen_vel_error().T)/self.gen_ad_factor()
        # Clip matrices for stability (optional, can be tuned or removed)
        self.k_mat = np.clip(self.k_mat, -100, 100)
        self.b_mat = np.clip(self.b_mat, -100, 100)
        return self.k_mat, self.b_mat
    
    def gen_pos_error(self):
        return self.q - self.q_d
    
    def gen_vel_error(self):
        return self.dq - self.dq_d
    
    def gen_track_error(self):
        return (self.k * self.gen_vel_error() + self.gen_pos_error())
    
    def gen_ad_factor(self):
        return self.a/(1.0 + self.b * la.norm(self.gen_track_error()) * la.norm(self.gen_track_error()))
    
    def calc_tau_fb(self):
        # self.fb_tau_mat = - self.k_mat * self.gen_pos_error() - self.b_mat * self.gen_vel_error()
        self.fb_tau_mat = self.k_mat * self.gen_pos_error() + self.b_mat * self.gen_vel_error()
        return self.fb_tau_mat
    
    def calc_tau_ff(self):
        self.ff_tau_mat = self.gen_ad_factor() * self.gen_track_error()
        return self.ff_tau_mat
    
    def get_tau(self):
        return self.calc_tau_fb() + self.calc_tau_ff()