import torch
import numpy as np
# import random
import taichi as ti
import datetime
import os
# import atexit
from pyevtk.hl import pointsToVTK
# import pandas as pd
import meshio
import time
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from TD3_BASE import TD3, ReplayBuffer
# from torchviz import make_dot
ti.init(debug=False, arch=ti.cuda, default_fp=ti.f64, packed = True, device_memory_fraction=0.9)
dim = 2 # 次元
dx = 5.0e1
inv_dx = 1 / dx
box_size = ti.Vector([7.0e3, 1.0e4]) # 計算領域のサイズ
area_start = ti.Vector([-3.5e3, -1.0e3]) # 計算領域の左下の座標
n_grid_x, n_grid_y = int(box_size.x * inv_dx + 1), int(box_size.y * inv_dx + 1)    #Euler点の数

BC1 = -1
BC2 = -2
OUTER_SHAPE = -3
gravity = False #重力を考慮するか
area_boundary = False #計算box内に収めるか
Dirichlet_Condition = True
Rienamm_metrics = False
self_colision = False

# alpha numbers
ACTUATOR1 = 0  # label of fiber domain
ACTUATOR2 = 1  # label of volume domain
ACTUATOR3 = 2
ACTUATOR4 = 3
ACTUATOR5 = 4
ACTUATOR6 = 5
n_actuators = 6


torch.set_printoptions(edgeitems=1000)

#graph setting
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams["font.size"] = 18
plt.rcParams['figure.figsize'] = [9, 7.5]
plt.rcParams["legend.edgecolor"] = 'black'  


E, nu1 = 1.0, -0.1 #物体１の力学特性
E2, nu2 = 1.0e2, 0.0 
la1, mu1 = E * nu1 / ((1+nu1) * (1-2*nu1)) , E / (2 * (1+nu1))
print(f"la:{la1}, mu:{mu1}")
la2, mu2 = E2 * nu2 / ((1+nu2) * (1-2*nu2)) , E2 / (2*(1+nu2))
print(f"la2:{la2}, mu2:{mu2}")
rho1 = 1.0e-2 # 密度
rho2 = 1.0
g = 9.8e-3 # 重力加速度
sound_s = ti.sqrt((la1 + 2 * mu1) / rho1)     # 縦波の速度
sound_s2 = ti.sqrt((la2 + 2* mu2) / rho2) 

# ここでクーラン条件を満たしているかを確認する
dt_max1 = 0.1 * dx / sound_s     # (臨海時間ステップ) / 10
dt_max2 = 0.1 * dx / sound_s2     # (臨海時間ステップ) / 10
dt_max = dt_max1
if dt_max2<dt_max1:
    dt_max = dt_max2

print("dt_max: ", dt_max)

dt = dt_max    # 時間の刻み幅
print("dt: ", dt)

snap = int(10/dt) #1sおきにvtdを出力する
print("Snap:", snap)

second = 100

num_steps = snap * second    # 計算回数(num_steps*dt=seconds?)
print("num_steps: ", num_steps)


# 出力ファイル保存先の指定
file_name = datetime.datetime.now().strftime('%Y_%m_%d'+'/%m_%d_%H_%M_%S')
folder_name = os.path.splitext(os.path.basename(__file__))[0]
dr_original = folder_name+'/'+file_name
os.makedirs(dr_original, exist_ok=True)
#'/'+str(inod)+str(hnod)+str(hnod2)+str(onod)+
# メッシュファイルの読み込み
msh1 = meshio.read('Tutorial2023/MPM/Tutorial/tutorial1/geometry/square2.msh')
msh1_size = [1.0e3, 5.0e3]
msh1_pos = [0.0, 2.5e3] # 初期位置の座標
initial_pos = []

# for i in range(len(initial_pos)):
#     if i%2 == 0:
#         initial_pos[i] = initial_pos[i]/100.0
#     else:initial_pos[i] = (initial_pos[i] - 2500.0)/(100.0)
# msh2 = meshio.read('Tutorial2023/MPM/Tutorial/tutorial2/geometry/rectangular1.msh')
# msh2_size = [1.0e3, 1.0e3, 1.0e3]
# msh2_pos = [-1.5e3, 0.0, 4.5e3] # 初期位置の座標

# n_faces_1, _ = msh1.cells_dict['triangle'].shape  # 要素数, 次数
# n_faces_2 = msh1.cells_dict['triangle'].shape[0]  # 要素数
# n_faces = n_faces_1 + n_faces_2

# index_1 = msh1.cells_dict['triangle']
# index_2 = msh2.cells_dict['triangle']
# index = np.append(index_1, index_2, axis=0)
# indices = np.reshape(index, n_faces*_)

@ti.data_oriented
class MPM():
    def __init__(self,msh1,dt,num_iter,num_steps, inod):
        self.dt = dt
        self.BIG = 1e32
        self.DIVERGE = ti.field(dtype=int, shape=())
        self.DIVERGE[None] = 0
        self.DIVERGE_SPOT = ti.field(dtype=int, shape=())
        self.DIVERGE_SPOT[None] = 0
        # self.OUTER_SHAPE_NUM = ti.field(dtype=int, shape=())
        # self.OUTER_SHAPE_NUM[None] = 0
        # self.n_actuators = n_actuators
        # self.n_sin_waves = n_sin_waves
        # self.actuation_omega = actuation_omega
        # self.act_strength = act_strength
        self.num_steps = num_steps
        # 読み込む物体の数によって変更が必要
        self.n_particles, _ = msh1.points.shape  # 要素数, 次数
        self.n_elements, self.n_vertices = msh1.cells_dict['triangle'].shape    # 要素数, 頂点数
        # self.done = False

        self.mass = ti.field(dtype=float, shape=self.n_particles)  # 各Lagrange点の質量
        self.x = ti.Vector.field(dim, dtype=float, shape=self.n_particles, needs_grad=True)   # 各Lagrange点の座標
        self.x_rest = ti.Vector.field(dim, dtype=float, shape=self.n_particles)
        self.v = ti.Vector.field(dim, dtype=float, shape=self.n_particles)    # 各Lagrange点の速度
        self.C = ti.Matrix.field(dim, dim, dtype=float, shape=self.n_particles)   # 各Lagrange点のアフィン速度
        self.grid_mass = ti.field(dtype=float, shape=(n_grid_x, n_grid_y))   # 各Euler点の質量
        self.grid_momentum = ti.Vector.field(dim, dtype=float, shape=(n_grid_x, n_grid_y))    # 各Euler点の運動量
        self.grid_domain = ti.field(dtype=int,shape=(dim, n_grid_y))
        self.domain = ti.field(dtype=int,shape=2*dim)        
        self.restT = ti.Matrix.field(dim, dim, dtype=float, shape=self.n_elements)
        self.element_energy = ti.field(dtype=float, shape=self.n_particles)
        self.total_energy = ti.field(dtype=float, shape=(), needs_grad=True)
        # self.kinetic_energy = ti.field(dtype=float, shape=())
        self.vertices = ti.field(dtype=ti.i32, shape=(self.n_elements, self.n_vertices))
        self.volumes = ti.field(dtype=float,shape=self.n_elements)
        self.bc = ti.field(dtype=float, shape=4)
        self.OUTER_SHAPE_pos = ti.field(dtype=float, shape=10)
        for i in range(10):
            self.OUTER_SHAPE_pos[i] = 500.0*(i+1)
        # self.OUTER_SHAPE_pos = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0]

        self.bool_on_bc = ti.field(dtype=int, shape=self.n_particles)
        # self.bool_on_Riemann = ti.field(dtype=int, shape=self.n_elements)
        self.grid_bc = ti.field(dtype=int, shape=(n_grid_x,)*2)
        self.epsilon = ti.Matrix.field(dim, dim, dtype=float, shape=self.n_elements)
        self.sigma = ti.Matrix.field(dim, dim, dtype=float, shape=self.n_elements)
        self.la = ti.field(dtype=float, shape=self.n_elements)
        self.mu = ti.field(dtype=float, shape=self.n_elements)
        self.Fp=ti.field(dtype=float, shape=self.n_elements)
        # self.Fpzz=ti.field(dtype=float, shape=self.n_elements)
        # self.x_cell=ti.field(dtype=float, shape=self.n_elements)
        # self.z_cell=ti.field(dtype=float, shape=self.n_elements)

        # self.s_cell=ti.field(dtype=float, shape=self.n_elements)
        
        # self.bool_on_surface = ti.field(dtype=int, shape=self.n_particles) 
        self.num_steps = ti.field(dtype=int, shape=())
        self.num_steps[None] = num_steps
        self.time_step = ti.field(dtype=int, shape=())
        self.time_step[None] = 1
        self.iter = ti.field(dtype=int, shape=())
        self.target_p = ti.field(dtype=int, shape=())
        self.distance = ti.field(dtype=float, shape=())
        self.pre_distance = ti.field(dtype=float, shape=())
        self.alpha_dum = ti.field(dtype=float, shape=())
        self.reward_area = ti.field(dtype=float, shape=())
        self.reward_area[None] = 600.0
        
        self.state = ti.Matrix.field(n=1, m=inod, dtype=float, shape=(num_steps+1))
        self.outer_state_p = ti.field(dtype=int, shape=20)
        # for i in range(44):
        #     self.outer_state[i] = initial_pos[i]/(500 + 4500(i%2))
        #     print(self.outer_state[i])
        self.reward_detail = ti.field(dtype=float, shape=(num_steps+1, 4))
        
        # self.w_in_h1 = ti.Matrix.field(n=hnod, m=inod, dtype=float, shape=())
        # self.w_h1_h2 = ti.Matrix.field(n=hnod2, m=hnod, dtype=float , shape=())
        # self.w_h2_out = ti.Matrix.field(n=onod, m=hnod2, dtype=float , shape=())
        
        # self.w_in_h1_prop = ti.Matrix.field(n=hnod, m=inod, dtype=float, shape=(num_steps+1))
        # self.w_h1_h2_prop = ti.Matrix.field(n=hnod2, m=hnod, dtype=float , shape=(num_steps+1))
        # self.w_h2_out_prop = ti.Matrix.field(n=onod, m=hnod2, dtype=float , shape=(num_steps+1))
        
        # self.w_in_h1_momemtum = ti.Matrix.field(n=hnod, m=inod, dtype=float, shape=(num_steps+1))
        # self.w_h1_h2_momentum = ti.Matrix.field(n=hnod2, m=hnod, dtype=float , shape=(num_steps+1))
        # self.w_h2_out_momentum = ti.Matrix.field(n=onod, m=hnod2, dtype=float , shape=(num_steps+1))

        # self.hid1_tanh = ti.Matrix.field(n=hnod, m=1, dtype=float , shape=(num_steps+1))
        # self.hid2_tanh = ti.Matrix.field(n=hnod2, m=1, dtype=float , shape=(num_steps+1))
        # self.out_tanh = ti.Matrix.field(n=onod, m=1, dtype=float , shape=(num_steps+1))
        # self.a1  = ti.field(dtype=float, shape=num_iter+1, needs_grad = True)
        # self.b1  = ti.field(dtype=float, shape=num_iter+1, needs_grad = True)
        self.reward_history = ti.field(dtype=float, shape=(num_iter+6,num_steps))
        self.total_reward = ti.field(dtype=float, shape=num_iter+6)
        self.goal = ti.Vector.field(dim, dtype=float, shape=())
        self.goal[None] = ti.Vector([-250.0, 5500.0])
        # self.pos_goal_his = ti.field(dtype=float, shape=(num_iter+1,2))

        # self.weights = ti.field(dtype=float, shape=(n_actuators, n_sin_waves), needs_grad = True)
        # self.bias = ti.field(dtype=float, shape=(n_actuators), needs_grad=True)
        self.actuation = ti.field(dtype=float, shape=(num_steps+1,n_actuators+1))
        for i in range(n_actuators +1):
            self.actuation[0,i] = 0.0
        self.bool_actuator_id = ti.field(dtype=int, shape=self.n_elements)


    def read_mesh(self, msh1):
        self.p = msh1.points
        self.x.from_numpy(self.p)
        self.p_ = msh1.cells_dict['triangle']    # 2つ目の物体の各点のID書き換え
        self.vertices.from_numpy(self.p_)
        # print(np.shape(self.vertices))
        self.initialize()


    def initialize(self):
        self.set_volumes()
        self.get_limitaion()
        self.compute_cell()
        self.set_boundary_points()
        # self.set_outer_shape()


    @ti.kernel
    def set_volumes(self):
        for i in range(self.n_particles):
            self.x[i].x = (self.x[i].x - msh1_size[0]/2 + msh1_pos[0])*0.5
            self.x[i].y = (self.x[i].y - msh1_size[1]/2 + msh1_pos[1])
            self.v[i] = [0.0, 0.0]   # 初速度
            self.x_rest[i] = self.x[i]

        for i in range(self.n_elements):
            self.restT[i] = self.compute_T(i)  # Compute rest T
            self.volumes[i] = self.restT[i].determinant() * 0.5
            self.mass[self.vertices[i, 0]] += rho1 * self.volumes[i] / self.n_vertices
            self.mass[self.vertices[i, 1]] += rho1 * self.volumes[i] / self.n_vertices
            self.mass[self.vertices[i, 2]] += rho1 * self.volumes[i] / self.n_vertices


    def get_limitaion(self):
        pos = self.x.to_numpy()
        y_low = pos[:, 1].min()
        y_high = pos[:, 1].max()
        x_low = pos[:,0].min()
        x_high = pos[:,0].max()
        bc = np.array([y_low, y_high, x_low, x_high])
        # print(bc)
        #[   0. 5000. -500.  500.]
        self.bc.from_numpy(bc)


    @ti.kernel
    def compute_cell(self):
        # self.bool_on_Riemann.fill(0)
        inv_z_dis = 1/(self.bc[3]-self.bc[2])
        inv_x_dis = 1/(self.bc[1]-self.bc[0])
        for i in range(self.n_elements):
            a = self.vertices[i, 0]
            b = self.vertices[i, 1]
            c = self.vertices[i, 2]
            z_max = ti.max(self.x[a].x,self.x[b].x,self.x[c].x)
            x_max = ti.max(self.x[a].y,self.x[b].y,self.x[c].y)
            z_dis =(z_max - self.bc[2])*inv_z_dis
            x_dis = (x_max - self.bc[0])*inv_x_dis
            self.bool_actuator_id[i] = -1 
            self.la[i] = la2
            self.mu[i] = mu2
            # #緩衝材あり
            # if x_dis < 0.33:                
            #     if z_dis < 0.5:
            #         self.bool_actuator_id[i] = ACTUATOR1
            #     elif z_dis > 0.6:
            #         self.bool_actuator_id[i] = ACTUATOR2
            # elif 0.35 < x_dis < 0.65:
            #     if z_dis < 0.5:
            #         self.bool_actuator_id[i] = ACTUATOR3
            #     elif z_dis > 0.6:
            #         self.bool_actuator_id[i] = ACTUATOR4
            # elif 0.67 < x_dis < 0.97:
            #     if z_dis < 0.5:
            #         self.bool_actuator_id[i] = ACTUATOR5
            #     elif z_dis > 0.6:
            #         self.bool_actuator_id[i] = ACTUATOR6   
            #緩衝材なし
            if x_dis < 0.33:                
                self.la[i] = la1
                self.mu[i] = mu1                
                if z_dis < 0.5:
                    self.bool_actuator_id[i] = ACTUATOR1
                else:
                    self.bool_actuator_id[i] = ACTUATOR2
            elif x_dis < 0.67:
                self.la[i] = la1
                self.mu[i] = mu1                
                if z_dis < 0.5:
                    self.bool_actuator_id[i] = ACTUATOR3
                else:
                    self.bool_actuator_id[i] = ACTUATOR4
            elif x_dis < 0.99:
                self.la[i] = la1
                self.mu[i] = mu1                
                if z_dis < 0.5:
                    self.bool_actuator_id[i] = ACTUATOR5
                else:
                    self.bool_actuator_id[i] = ACTUATOR6     
            # x = self.a1[self.iter[None]]*3.14*(z_max - self.bc[2])*inv_z_dis
            # z_dis = ti.cos(x)
            # x_dis = self.b1[self.iter[None]] - (x_max - self.bc[0])*inv_x_dis
            # self.s_cell[i] = z_dis*self.b1[self.iter[None]]
        # print("a1:", self.a1[self.iter[None]])
        # print("b1:", self.b1[self.iter[None]])

                # print(self.s_cell[i])ok
                # if self.s_cell[i] < 0.5:
                #     self.bool_on_Riemann[i] = ALPHA1
                # else:
                #     self.bool_on_Riemann[i] = ALPHA2 
                # print(self.bool_on_Riemann[i])ok

    @ti.kernel
    def set_boundary_points(self):
        # set boundary and surface points
        
        for p in range(self.n_particles):
            self.bool_on_bc[p] = 0
            if Dirichlet_Condition:
                if abs(self.x[p].y-self.bc[0]) < 1e-6:
                    self.bool_on_bc[p] = BC1
                elif ((abs(self.x[p].x - self.bc[2]) < 1e-6)):
                    for i in range(10):
                        if abs(self.x[p].y - self.OUTER_SHAPE_pos[i]) < 48:
        
                            self.bool_on_bc[p] = OUTER_SHAPE

                            self.outer_state_p[i] = p
                            # print(self.x[p].x, self.x[p].y)

                            
                            # break
                            # print(count_outer_shape_num)
                elif (abs(self.x[p].x - self.bc[3]) < 1e-6):
                    for j in range(10):
                        if abs(self.x[p].y - self.OUTER_SHAPE_pos[j]) < 48:
                            self.bool_on_bc[p] = OUTER_SHAPE
                            self.outer_state_p[10+j] = p
                            # print(self.x[p].x, self.x[p].y)
                            
                            # break
                    

                    
                elif abs(self.x[p].y-self.bc[1]) < 1e-6:
                    if (abs(self.x[p].x) < 20) :
                        self.bool_on_bc[p] = BC2
                        self.target_p[None] = p
                        self.state[0][0,0] = (self.x[p].y - self.bc[1]+2000.0)/4000.0
                        self.state[0][0,1] = (self.x[p].x+1000.0)/2000.0
                        # print(self.x[p].y)
                        # print(self.state[0][0,0])
                        # self.state[None][0,1] = self.v[p].x
                        self.state[0][0,2] = 0.5
                        self.state[0][0,3] = 0.5
                        # self.state[None][0,2] = 0/self.num_steps[None]
                        self.state[0][0,4] = (self.x[p].y - self.goal[None].y+3000.0)/6000.0
                        self.state[0][0,5] = (self.x[p].x - self.goal[None].x+1500.0)/3000.0
                        # print(self.state[0]) 
        # print(self.outer_state)               

                
    def set_outer_shape(self):
        x = np.array(initial_pos)
        self.outer_state.from_numpy(x)
        # print(self.outer_state)
                
            
    @ti.func
    def compute_T(self, i):
        a = self.vertices[i, 0]
        b = self.vertices[i, 1]
        c = self.vertices[i, 2]
        ab = self.x[b] - self.x[a]
        ac = self.x[c] - self.x[a]
        return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])


    @ti.kernel
    def compute_total_energy(self):
        # print("act:",self.actuation[self.time_step[None]])
        
        for i in range(self.n_elements):
            
            self.Fp[i] = self.actuation[self.time_step[None], self.bool_actuator_id[i]]
            f = self.actuation[self.time_step[None], self.bool_actuator_id[i]]
            # self.check_diverge(f, 20)
            Fp = ti.Matrix([[1.0,0.0],[0.0, 1.0+f]])
            # elif self.bool_on_Riemann[i] == ALPHA2:
            #     Fp = ti.Matrix([[1,0,0],[0,1,0],[0,0,1-self.Fp]])
            currentT = self.compute_T(i)
            # self.check_diverge(any(currentT), 7)
            F = currentT @ self.restT[i].inverse()
            # self.check_diverge(any(F), 8)
            C_ = F.transpose() @ F
            # self.check_diverge(any(C_), 9)
            Cp = Fp.transpose() @ Fp
            # self.check_diverge(any(Cp), 10)
            E_ = 0.5*(C_-Cp)
            # self.check_diverge(any(E_), 11)
            self.epsilon[i] = E_
            
            # NeoHookean
            Fe = F@Fp.inverse()
            # self.check_diverge(any(Fe), 12)            
            I1 = (Fe @ Fe.transpose()).trace()
            # self.check_diverge(I1, 13)            
            J = Fe.determinant()   
            # self.check_diverge(J, 14)            
            self.element_energy[i] = 0.5 * self.mu[i] * (I1 - dim) - self.mu[i] * ti.log(J) + 0.5 * self.la[i] * ti.log(J)**2
            # self.check_diverge(self.element_energy[i], 5)
            self.total_energy[None] += self.element_energy[i] * self.volumes[i]

            #St.Venant-Kirichhof
            # Cpinv = Cp.inverse()
            # E1 = E_ @ Cpinv
            # E2 = E1 @ E1
            # self.total_energy[None] += \
            #     (la1/2*E1.trace()**2 + mu1*E2.trace()) * Fp.determinant() * self.volumes[i]

            # S_ = la1*E1.trace()*Cpinv + 2*mu1*Cpinv @ E_ @ Cpinv
            # J_ = F.determinant() / Fp.determinant()
            # self.sigma[i] = (F @ S_ @ F) / J_
            # self.sigma[i] =la1*E1.trace()*Cpinv + 2*mu1*E_ @ Cpinv


            # else:
            #     currentT = self.compute_T(i)
            #     F = currentT @ self.restT[i].inverse()
            #     Fp = ti.Matrix([[1,0,0],[0,1,0],[0,0,1]])
            #     C_ = F.transpose() @ F
            #     Cp = Fp.transpose() @ Fp
            #     E_ = 0.5*(C_-Cp)
            #     self.epsilon[i] = E_
            #     # NeoHookean
            #     I1 = (F @ F.transpose()).trace()
            #     J = F.determinant()
            #     element_energy = 0.5 * mu2 * (I1 - dim) - mu2 * ti.log(J) + 0.5 * la2 * ti.log(J)**2
            #     self.total_energy[None] += element_energy * self.volumes[i]

    @ti.func
    def check_diverge(self, test_num, num_spot):
        if not (test_num < self.BIG) :
            if self.DIVERGE[None] == 0:
                self.DIVERGE_SPOT[None] = num_spot
            self.DIVERGE[None] = 1

    @ti.kernel
    def calculate_alpha_dum(self):
        uKu, uMu= 0.0, 0.0
        for p in range(self.n_particles):
            u_this = self.x[p] - self.x_rest[p]
            # if not all(u_this < self.BIG) :
            #     if self.DIVERGE[None] == 0:
            #         self.DIVERGE_SPOT[None] = 0
            #     self.DIVERGE[None] = 1

            f_p_int = -self.x.grad[p]
            # if not all(f_p_int < self.BIG) :
            #     if self.DIVERGE[None] == 0:
            #         self.DIVERGE_SPOT[None] = 1
            #     self.DIVERGE[None] = 1
            #     continue

            uMu += self.mass[p]*u_this.norm_sqr()
            
            _uKu = abs(u_this.dot(f_p_int))
            # if not (_uKu < self.BIG) :
            #     if self.DIVERGE[None] == 0:
            #         self.DIVERGE_SPOT[None] = 2
            #     self.DIVERGE[None] = 1
            #     continue
            uKu += _uKu
            
            # if not (uMu < self.BIG) :
            #     if self.DIVERGE[None] == 0:
            #         self.DIVERGE_SPOT[None] = 3
            #     self.DIVERGE[None] = 1
        # if uKu<0:
        #     print(self.time_step[None],uKu, uMu)
        self.alpha_dum[None] = 2*ti.sqrt(abs(uKu)/uMu) if uMu > 1.0e-5 else 0.0
        # if 0.0 < self.alpha_dum[None] < 0.1:
        #     self.alpha_dum[None] = 0.1
        # if not (self.alpha_dum[None] < self.BIG):
        #     if self.DIVERGE[None] == 0:
        #         self.DIVERGE_SPOT[None] = 4
        #     self.DIVERGE[None] = 1


        
    @ti.kernel
    def p2g(self):
        for p in range(self.n_particles):
            # self.alpha_dum[None] = 5.0
            # if self.bool_on_bc[p] != BC1:
            base = ti.cast((self.x[p] - area_start) * inv_dx - 0.5, ti.i32) # 基準とするEuler点の位置/dx
            fx = (self.x[p] - area_start) * inv_dx - ti.cast(base, float)   # Lagrange点の位置/dx
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            beta = 0.5*dt*self.alpha_dum[None]
            f_p_int = -self.x.grad[p]#内力
            # print(f_p_int, self.v[p])
            # if self.mass[p] > 0:
            #     self.v[p] += dt*f_p_int/self.mass[p] - dt*self.alpha_dum[None]*self.v[p]#次元おかしい気がする
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    I = ti.Vector([i, j])
                    dpos = (float(I) - fx) * dx
                    weight = w[i].x * w[j].y 
                    self.grid_mass[base + I] += weight * self.mass[p]
                    # if not all(self.grid_mass[base + I] < self.BIG) :
                    #     print("nan0", p)
                    #     self.DIVERGE[None] = 1
                        # quit()
                    # print(p, self.mass[p] * (self.v[p] + self.C[p] @ dpos)*(1-dt*self.alpha_dum[None]))
                    self.grid_momentum[base + I] += weight * (((1-beta)*self.mass[p] * self.v[p] + dt * f_p_int ) / (1+beta) + self.mass[p] * self.C[p] @ dpos)
                    if self.bool_on_bc[p % self.n_particles] == BC1:
                        self.grid_bc[base + I] = self.bool_on_bc[p % self.n_particles]
                    for _ in ti.static(range(dim)):
                        self.grid_domain[_,base[_]+I[_]] = 1
    @ti.kernel
    def set_domain(self):
        for i in ti.static(range(dim)):
            left_=0
            right_=n_grid_y-1
            while self.grid_domain[i,left_+3] == 0:
                left_+=3
            
            while self.grid_domain[i,right_ - 3] == 0:
                right_ -= 3
            
            self.domain[2*i] = left_
            self.domain[2*i +1] = right_
            # print(i, left_, right_)
            
    @ti.kernel
    def grid_op(self):
        for i, j in ti.ndrange((self.domain[0],self.domain[1]),(self.domain[2],self.domain[3])):
            if self.grid_mass[i, j] > 0:
                inv_m = 1 / self.grid_mass[i, j]
                self.grid_momentum[i, j] = inv_m * self.grid_momentum[i, j]#ここで速度になってる？
            self.grid_momentum[i, j] = 0.0 if self.grid_mass[i, j] == 0.0 else self.grid_momentum[i, j]
            self.check_diverge(any(self.grid_momentum[i,j]), 6)
            # if i, j, k == 0, 0,0:
            #     print(self.grid_momentum[i,j,k])
            if gravity:
                self.grid_momentum[i, j].y += dt*g*10.0 

            if area_boundary:
                boundary = 3 #num of grid from the boundary area
                if i < boundary and self.grid_momentum[i, j].x < 0.0:
                    self.grid_momentum[i, j].x = 0.0
                if i > n_grid_x - boundary and self.grid_momentum[i, j].x > 0.0:
                    self.grid_momentum[i, j].x = 0.0
                if j < boundary and self.grid_momentum[i, j].y < 0.0:
                    self.grid_momentum[i, j].y = 0.0
                if j > n_grid_y - boundary and self.grid_momentum[i, j].y > 0.0:
                    self.grid_momentum[i, j].y = 0.0

            #direchret 
            I = ti.Vector([i, j])
            if self.grid_bc[I] == BC1:
                # print("set bc")
                self.grid_momentum[I] = [0.0, 0.0]

# 一般的にはgrid_opで境界条件より３particle分grid_vを０にすることでディレクレ境界条件を作る。


    @ti.kernel
    def g2p(self):
        # self.kinetic_energy[None] = 0
        # indices = 1
        for p in self.x:
            base = ti.cast((self.x[p] - area_start) * inv_dx - 0.5, ti.i32)
            fx = (self.x[p] - area_start) * inv_dx - float(base)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector([0.0, 0.0])
            new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            
            # if self.bool_on_bc[p] != BC1:
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    I = ti.Vector([i, j])
                    dpos = (float(I) - fx) * dx
                    g_v = self.grid_momentum[base + I]
                    weight = w[i].x * w[j].y 
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx**2
            self.v[p] = new_v 

            # if self_colision:
                
                # print("state[None]:", self.state[None][1], self.state[None][3])
            self.x[p] += dt * self.v[p]
            self.C[p] = new_C
            
            # self.kinetic_energy[None] += 0.5*self.mass[p]*(ti.pow(self.v[p].x, 2) + ti.pow(self.v[p].y, 2))
            
            # if self.bool_on_bc[p] == BC2:
            # self.state[None][0,0] = self.x[p].x
        self.state[self.time_step[None]][0,0] = (self.x[self.target_p[None]].y - self.bc[1]+2000.0)/4000.0
        self.state[self.time_step[None]][0,1] = (self.x[self.target_p[None]].x+1000.0)/2000.0
        # print(self.x[self.target_p[None]].y)
        # print(self.state[self.time_step[None]][0,0])
        # self.state[None][0,1] = self.v[self.target_p[None]].x
        self.state[self.time_step[None]][0,2] = (self.v[self.target_p[None]].y+15.0)/30.0
        self.state[self.time_step[None]][0,3] = (self.v[self.target_p[None]].x+15.0)/30.0
        # self.state[None][0,2] = self.time_step[None]/self.num_steps[None]
        self.state[self.time_step[None]][0,4] = (self.x[self.target_p[None]].y-self.goal[None].y+3000.0)/6000.0
        self.state[self.time_step[None]][0,5] = (self.x[self.target_p[None]].x-self.goal[None].x+1500.0)/3000.0
        # self.check_diverge(any(self.state[self.time_step[None]]), 21)
        # print(self.state[None][0,3])
        self.distance[None] = ti.pow(self.x[self.target_p[None]].x - self.goal[None].x, 2) + ti.pow(self.x[self.target_p[None]].y - self.goal[None].y, 2)
            # if self.bool_on_bc[p] == OUTER_SHAPE:
            #     self.outer_state[indices*2-1] = self.x[p].x/1000.0
            #     self.outer_state[indices*2] = (self.x[p].y-2500.0)/1000.0
            #     indices += 1
                
                
    @ti.kernel
    def compute_reward(self):
        # reward = -1.0
        # reward = -2e-5*self.distance[None]
        
        #目的地との距離の変化量の報酬
        reward = (self.pre_distance[None] - ti.sqrt(self.distance[None]))
        # if self.distance[None] < self.pre_distance[None]:
        #     reward = 1.0
        self.pre_distance[None] = ti.sqrt(self.distance[None])
        self.reward_detail[self.time_step[None], 0] = reward
        # print(reward)
        #速度のー報酬
        # _reward = 0.0
        # _reward -= 0.05*-ti.pow(self.state[self.time_step[None]][0,2] , 2)
        # _reward -= 0.05*ti.pow(self.state[self.time_step[None]][0,3] , 2)
        # self.reward_detail[self.time_step[None], 1] = _reward
        # reward+=_reward

        # self.reward_detail[self.time_step[None], 1] = 20.0*ti.pow(self.state[self.time_step[None]][0,1], 2)  
        
        #リーマン計量の変化量のー報酬
        # _reward = 0.0
        # for i in range(n_actuators):
        #     _reward -= 3.0*ti.pow(self.actuation[self.time_step[None], i]-self.actuation[self.time_step[None]-1, i], 2)
        # self.reward_detail[self.time_step[None], 2] = _reward
        # reward += _reward
        
        #-energy reward
        # for i in range(20):
        #     reward -= 2e-4*self.element_energy[self.outer_state_p[i]]

        #リーマン計量のー報酬
        # _reward = 0.0
        # # _reward -= 1.0e-11*(self.total_energy[None] + self.kinetic_energy[None])
        # for i in range(n_actuators):
        #     _reward -= 0.1*ti.pow(self.actuation[self.time_step[None], i], 2)
        # self.reward_detail[self.time_step[None], 1] = _reward
        # reward += _reward
        
        #目的地に近かったらボーナス
        if ti.sqrt(self.distance[None]) < self.reward_area[None]:
            reward += 2.0*(1.0-self.distance[None]/ti.pow(self.reward_area[None],2))
        self.reward_detail[self.time_step[None], 1] = reward  

        self.reward_history[self.iter[None], self.time_step[None]] = reward
        self.total_reward[self.iter[None]] += reward
        # self.check_diverge(reward, 22)


    def export_vtk(self, file_name):
        points_output = np.zeros((self.n_particles, 3), dtype=np.float64)
        points=self.x.to_numpy()
        points_output[:,0] = points[:,0]
        points_output[:,1] = points[:,1]
        triangle=self.vertices.to_numpy()
        cells=[
            ('triangle',triangle),
        ]
        mesh_=meshio.Mesh(
            points_output,
            cells,
            point_data={
                'boundary':self.bool_on_bc.to_numpy(),
                'energy':self.element_energy.to_numpy()
            },
            cell_data={
                'strain':[self.epsilon.to_numpy().reshape(-1,4)],
                'sigma':[self.sigma.to_numpy().reshape(-1,4)],
                'Riemann':[self.bool_actuator_id.to_numpy()],
                'Fp':[self.Fp.to_numpy()]
            }
        )
        mesh_.write(file_name)

    def export_goal(self, file_name):
        point_data={"goal": np.array([1])}
        pointsToVTK(file_name, np.array([self.goal[None].x]), np.array([self.goal[None].y]), np.array([1.0]), data=point_data)               

            
    def export_graph(self, file_name):
        # act = self.actuation.to_numpy()
        # ymax = act[1:].max()
        # ymin = act[1:].min()
        y = self.actuation.to_numpy().T
        fig = plt.figure(figsize=(12, 8))
        plt.title("actuation iter{0}, goal{1}\n".format(self.iter[None], self.goal[None]), fontsize=20)
        plt.ylabel("actuation", fontsize=20)
        plt.xlabel("time step", fontsize=20)
        for i, array in enumerate(y):
            if i == 6:
                break
            plt.plot(array, label=f"actuator #{i+1}")
        plt.grid()
        plt.legend(fontsize=15 ,loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        # plt.xlim(1,self.num_steps[None])
        # plt.ylim(ymin*0.95, ymax*1.05)
        fig.savefig(file_name+f'/{self.iter[None]}actuation.png')
        plt.close()

    @ti.func
    def _grid_clear(self,I):
        self.grid_mass[I]=0
        self.grid_momentum[I]=[0,0]


    @ti.kernel
    def grid_clear(self):
        for i,j in ti.ndrange(
            (self.domain[0],self.domain[1]),
            (self.domain[2],self.domain[3])
            ):
            I=ti.Vector([i,j])
            self._grid_clear(I)


    def clear(self):
        self.grid_clear()
        self.grid_bc.fill(0)
        self.grid_domain.fill(0)
        self.domain.fill(0)
        self.total_energy[None] = 0.0

    def set_grad(self):
        with ti.Tape(self.total_energy):
            self.compute_total_energy()

    def step(self):
        self.clear()
        self.set_grad()
        self.calculate_alpha_dum()    
        self.p2g()
        self.set_domain()
        self.grid_op()
        self.g2p()    
        # self.compute_reward()

    def reset(self):
        self.epsilon.fill(0)
        self.sigma.fill(0)
        self.Fp.fill(0)
        # self.x.fill(0)
        # self.mass.fill(0)
        self.v.fill(0)
        # self.Fpzz.fill(0)
        self.C.fill(0)
        self.element_energy.fill(0)
        self.total_energy[None] = 0
        self.time_step[None] = 1
        for i in range(n_actuators + 1):
            self.actuation[0,i] = 0.0
        for i in range(self.n_particles):
            # self.v[i] = [0.0, 0.0]   # 初速度
            self.x[i] = self.x_rest[i]
        self.pre_distance[None] = ti.sqrt(ti.pow(self.goal[None].x, 2) + ti.pow(self.goal[None].y -self.x[self.target_p[None]].y, 2))
        self.state[0][0,4] = (self.x[self.target_p[None]].y - self.goal[None].y+3000.0)/6000.0
        self.state[0][0,5] = (self.x[self.target_p[None]].x - self.goal[None].x+1500.0)/3000.0

        # print(self.pre_distance[None])
        # self.OUTER_SHAPE_NUM[None] = 0

    # @atexit.register
    def export_program(self):
        with open(__file__, mode="r") as fr:
                prog = fr.read()
        with open(dr_original+"/program.py", mode="w") as fw:
            fw.write(prog)
            fw.flush()
        with open("TD3_BASE.py", mode="r") as fr:
                prog = fr.read()
        with open(dr_original+"/td3_program.py", mode="w") as fw:
            fw.write(prog)
            fw.flush()

def gauss(x):
    if x < 10.0:
        return 10.0
    else: return (np.exp(-x**2 / (2.0*250**2)) -1.0)

def normalize(l):
    l_min = min(l)
    l_max = max(l)
    return [(i - l_min) / (l_max - l_min) for i in l]

# エージェントの初期化
STATE_DIM  = 48 # 状態の次元数
ACTION_DIM = 6  # 行動の次元数
MAX_ACTION = 0.7

goal_length_x = 500.0
goal_length_y = 1000.0
goal_center_y = 5000.0
learning_goal_x = np.linspace(-goal_length_x, goal_length_x, 5)
learning_goal_y = np.linspace(goal_center_y-goal_length_y, goal_center_y+goal_length_y, 5)
goal = [[x, y] for x in learning_goal_x for y in learning_goal_y]
test_num = len(goal)
# print("learning goal num:",learning_goal_num)

# large_R=5000.0
# small_R=1500.0
# # average_dis = np.sqrt((5000-2*np.sqrt(2)*(large_R-small_R)/np.pi)**2 + ((1-1/np.sqrt(2))(large_R-small_R)/np.pi)**2)
# test_R = np.linspace(small_R, large_R, 5)
# test_theta = np.linspace(np.pi*0.25, np.pi*0.75, 5)
# goal = [[round(R * np.cos(THETA), 1), round(R * np.sin(THETA), 1)] for R in test_R for THETA in test_theta]
# test_num = len(goal)

# TIME_START = time.time()
NUM_ITERS = 4000
WARMUP = np.min([NUM_ITERS//20, 1000])
OUTPUT_SPAN = np.max([NUM_ITERS//2, 100])
ACTUATION_OUTPUT_SPAN = np.max([NUM_ITERS//50, 10])
POLICY_OUTPUT_SPAN = np.max([NUM_ITERS//20, 10])
ACTUATION_SPAN = snap 
LR_DECAY_SPAN = (NUM_ITERS//6)
# MULTIPLE_TARGET = True
TEST_ONLY = False
LOAD_POLICY = True

def main():
    rng = np.random.default_rng()
    # FIRST_TRY = True
    # LAST_ITER = 0
    # num_goal = 1
    # iter_per_goal =1
    # iter_per_goal_history = []    
    # first_try = []
    accuracy_history = []
    reward_history = []
    reward_moving_ave = []  
    dis_moving_ave = []  
    moving_average = 75.0
    expl_noise = 0.1
    ACTUATION_CLIP = 0.15
    alr =1e-5
    clr=5e-5
    # DIVERGED = 0
    # a = 1.0
    
    mpm = MPM(msh1=msh1,dt=dt,num_iter=NUM_ITERS,num_steps=num_steps, inod=6)
    td3 = TD3(STATE_DIM, ACTION_DIM, MAX_ACTION, num_steps=num_steps, alr = alr, clr=clr)
    replay_buffer = ReplayBuffer(STATE_DIM, ACTION_DIM)
    
    if LOAD_POLICY:
        print("loading policy...")
        file_name = "2d_td3/2023_12_01/12_01_19_25_08/policy/iter16000"
        td3.load_critic(file_name)
        td3.load_actor(file_name)
        print("loading completed!")
        
    print(td3.actor)
    print(td3.critic)
    with open(dr_original+"/parameters.txt", mode="a") as fw:
        fw.write(f'\n{td3.actor}\n{td3.critic}')
        fw.close()      
    mpm.iter[None] = 0 
    mpm.read_mesh(msh1=msh1)
    mpm.export_program()

    if not TEST_ONLY :
        actuation_file = dr_original + '/actuation'
        os.makedirs(actuation_file, exist_ok=True)
        good_result_file = dr_original + '/good_results'
        os.makedirs(good_result_file, exist_ok=True)
        # csv_file = dr_original + '/csv'
        # os.makedirs(csv_file, exist_ok=True)
        
        while mpm.iter[None] <= NUM_ITERS:
            # print("replay buffer size:", replay_buffer.size)
            
            if mpm.iter[None] == WARMUP:
                iter_time_start = time.time()
            # print("before",mpm.actuation[0,0])
            mpm.reset()
            # print("outer shape point directory:",mpm.outer_state_p)
            # print("target point directory:",mpm.target_p)
            # print("after",mpm.actuation[0,0])
            # position = []
            # total_energy =[]
            # alpha_his = []      
            x_pos = []
            z_pos = []        
            # v_x = []
            # v_z = []
            state = []
            action = []         
            reward_per_span = []    
            
            # clip_min = -MAX_ACTION#*1e-2
            # clip_max = MAX_ACTION#*1e-2
            
            if mpm.iter[None] % OUTPUT_SPAN== 0 or mpm.iter[None] == NUM_ITERS -1:
                dr = dr_original + '/iter{:03d}'.format(mpm.iter[None])
                os.makedirs(dr, exist_ok=True)
                mpm.export_vtk(dr+'/iter{0}_timestep{1:05d}.vtu'.format(mpm.iter[None], 0))
                mpm.export_goal(dr+f'/iter{mpm.iter[None]}goal')
                with open(dr_original+"/parameters.txt", mode="a") as fw:
                    fw.write(f'\niter:{mpm.iter[None]}\n' + f'{list(td3.actor.parameters())}')
                    fw.write(f'{list(td3.critic.parameters())}')
                    fw.close()          

            if mpm.iter[None] % OUTPUT_SPAN== 1 or mpm.iter[None] == NUM_ITERS:
                dr = dr_original + '/iter{:03d}'.format(mpm.iter[None])
                os.makedirs(dr, exist_ok=True)
                mpm.export_vtk(dr+'/iter{0}_timestep{1:05d}.vtu'.format(mpm.iter[None], 0))
                mpm.export_goal(dr+f'/iter{mpm.iter[None]}goal')

            if mpm.iter[None] == 0:
                policy_file = dr_original + "/policy"
                os.makedirs(policy_file, exist_ok=True)
                with open(dr_original+"/parameters.txt", mode="a") as fw:
                    fw.write(f'\nalr:{alr}, clr:{clr}')
                    fw.close()        

            while mpm.time_step[None] <= num_steps:
                for i in range(n_actuators):
                    mpm.actuation[mpm.time_step[None], i] = mpm.actuation[mpm.time_step[None]-1, i]
                if mpm.time_step[None] % ACTUATION_SPAN == 1:
                    if mpm.time_step[None] != 1:
                        pre_state = []
                        pre_state = state
                        state = []
                        for i in range(6):
                            state.append(mpm.state[mpm.time_step[None]-1][0,i])
                        # for i in range(6):
                        #     state.append(mpm.actuation[mpm.time_step[None]-ACTUATION_SPAN,i])
                        for i in range(20):
                            state.append((mpm.x[mpm.outer_state_p[i]].x+1000.0)/2000.0)
                            state.append((mpm.x[mpm.outer_state_p[i]].y)/7000.0)
                        # for i in range(dim):
                        state.append((mpm.goal[None][0]+1000.0)/2000.0)
                        state.append((mpm.goal[None][1]-3000.0)/4000.0)
                        # state = 10.0*state
                
                        # reward = []
                        # _reward = -1.0
                        # if (state[4]**2 + state[5]**2) < (pre_state[4]**2 + pre_state[5]**2):
                        #     _reward = 1.0
                        _reward = 0.0

                        for i in range(ACTUATION_SPAN):
                            _reward += mpm.reward_history[mpm.iter[None], mpm.time_step[None]-ACTUATION_SPAN+i]
                        reward_per_span.append(_reward)

                        # reward.append(_reward)
                        replay_buffer.add(pre_state, action, state, _reward, 0)
                    else:
                        # state = []
                        for i in range(6):
                            state.append(mpm.state[0][0,i])
                        # for i in range(6):
                        #     state.append(mpm.actuation[0,i])
                            
                        for i in range(20):
                            state.append((mpm.x[mpm.outer_state_p[i]].x+1000.0)/2000.0)
                            state.append((mpm.x[mpm.outer_state_p[i]].y)/7000.0)
                        # for i in range(dim):
                        state.append((mpm.goal[None][0]+1000.0)/2000.0)
                        state.append((mpm.goal[None][1]-3000.0)/4000.0)
                        # state = 10.0*state
                    
                    action = td3.select_action(np.array(state))
                    action = (action+ rng.normal(0, MAX_ACTION * expl_noise, size=ACTION_DIM)).clip(-MAX_ACTION, MAX_ACTION)
                    # action=[0.45, -0.45, 0.45, 0.45, 0.45, -0.45]
                    for i in range(6):
                        action[i] = np.clip(action[i], mpm.actuation[mpm.time_step[None]-1, i]-ACTUATION_CLIP, mpm.actuation[mpm.time_step[None]-1, i]+ACTUATION_CLIP)
                    for i in range(n_actuators):
                        mpm.actuation[mpm.time_step[None], i] = action[i]
                mpm.step()
                # total_energy.append(mpm.total_energy[None])
                
                _reward = gauss(ti.sqrt(mpm.distance[None]))
                mpm.reward_detail[mpm.time_step[None], 0] = _reward  
                reward = _reward
                
                # _reward = 2.0*gauss(abs(mpm.x[mpm.target_p[None]].x-mpm.goal[None].x))
                # mpm.reward_detail[mpm.time_step[None], 1] = _reward  
                # reward += _reward
                
                _reward = -0.1*sum([i**2 for i in action])
                mpm.reward_detail[mpm.time_step[None], 1] = _reward  
                reward += _reward
                
                # _reward = 3.0*(mpm.pre_distance[None] - ti.sqrt(mpm.distance[None]))
                # mpm.pre_distance[None] = ti.sqrt(mpm.distance[None])
                # mpm.reward_detail[mpm.time_step[None], 2] = _reward
                # reward += _reward
                
                _reward = -mpm.total_energy[None]*1e-7
                mpm.reward_detail[mpm.time_step[None], 2] = _reward
                reward += _reward
                
                # _reward = 0.0
                # for i in range(n_actuators):
                #     _reward += -2.0*np.power(mpm.actuation[mpm.time_step[None], i]-mpm.actuation[mpm.time_step[None]-1, i], 2)
                # mpm.reward_detail[mpm.time_step[None], 4] = _reward  
                # reward += _reward

                mpm.reward_detail[mpm.time_step[None], 3] = reward  
                
                mpm.reward_history[mpm.iter[None], mpm.time_step[None]] = reward
                mpm.total_reward[mpm.iter[None]] += reward
                        
                x_pos.append(mpm.x[mpm.target_p[None]].x)
                z_pos.append(mpm.x[mpm.target_p[None]].y)
                # v_x.append(mpm.state[mpm.time_step[None]][0,3])
                # v_z.append(mpm.state[mpm.time_step[None]][0,2])
                
                if mpm.iter[None] % OUTPUT_SPAN == 0 or mpm.iter[None] == NUM_ITERS -1:
                    if mpm.time_step[None] % snap == 0:
                        mpm.export_vtk(dr+'/iter{0}_timestep{1:05d}.vtu'.format(mpm.iter[None],mpm.time_step[None]))
                if mpm.iter[None] % OUTPUT_SPAN == 1 or mpm.iter[None] == NUM_ITERS:
                    if mpm.time_step[None] % snap == 0:
                        mpm.export_vtk(dr+'/iter{0}_timestep{1:05d}.vtu'.format(mpm.iter[None],mpm.time_step[None]))
                
                

                if mpm.time_step[None] == num_steps:
                    pre_state = []
                    pre_state = state
                    state = []
                    for i in range(6):
                        state.append(mpm.state[mpm.time_step[None]][0,i])
                    # for i in range(6):
                    #     state.append(mpm.actuation[mpm.time_step[None]-ACTUATION_SPAN+1,i])
                    for i in range(20):
                        state.append((mpm.x[mpm.outer_state_p[i]].x+1000.0)/2000.0)
                        state.append((mpm.x[mpm.outer_state_p[i]].y)/7000.0)
                    # for i in range(dim):
                    state.append((mpm.goal[None][0]+1000.0)/2000.0)
                    state.append((mpm.goal[None][1]-3000.0)/4000.0)
                    # state = 10.0*state
                    
                    _reward = 0.0
                    for i in range(ACTUATION_SPAN):
                        _reward += mpm.reward_history[mpm.iter[None], mpm.time_step[None]-ACTUATION_SPAN+1+i]
                    reward_per_span.append(_reward)
                    # _reward = -1.0
                    # if (state[4]**2 + state[5]**2) < (pre_state[4]**2 + pre_state[5]**2):
                    #     _reward = 1.0 

                    replay_buffer.add(pre_state, action, state, _reward, 1)
                    # make_dot(action, params=list(td3.actor.parameters()), show_attrs=True, show_saved=True).render("rnn_torchviz", format="png")
                # move to after 1 iter?
                # if mpm.time_step[None]%1000 == 0:
                #     print(state) 
                mpm.time_step[None] += 1

            if mpm.iter[None]>=WARMUP:
                for i in range(num_steps):
                    td3.train(replay_buffer)
            # for i in range(4):
            #     print(mpm.reward_detail[1, i])

        
            accuracy = np.sqrt(mpm.distance[None])
            # if mpm.iter[None] % int(np.max([NUM_ITERS/2000, 1])) ==0:
            accuracy_history.append(accuracy)
            reward_history.append(mpm.total_reward[mpm.iter[None]])
            if mpm.iter[None] >= moving_average:
                _ave=0.0
                for i in range(int(moving_average)):
                    _ave += reward_history[-(i+1)]
                reward_moving_ave.append(_ave/moving_average)
                _ave=0.0
                for i in range(int(moving_average)):
                    _ave += accuracy_history[-(i+1)]
                dis_moving_ave.append(_ave/moving_average)
            else:
                _ave=0.0
                for i in range(mpm.iter[None]+1):
                    _ave += reward_history[-(i+1)]
                reward_moving_ave.append(_ave/float(mpm.iter[None]+1))
                _ave=0.0
                for i in range(mpm.iter[None]+1):
                    _ave += accuracy_history[-(i+1)]
                dis_moving_ave.append(_ave/float(mpm.iter[None]+1))                

            print(f'iter:{mpm.iter[None]}'\
                , f'goal:{mpm.goal[None]}'\
                , f'finalpos:[{mpm.x[mpm.target_p[None]].x:.1f} {mpm.x[mpm.target_p[None]].y:.1f}]' \
                , f'reward:{mpm.total_reward[mpm.iter[None]]:.1f}'\
                , f'finaldis:{np.sqrt(mpm.distance[None]):.1f}')
                # , f'accuracy:{accuracy:.1f}%')

            with open(dr_original+"/results.txt", mode="a") as fw:
                fw.write(f'iter:{mpm.iter[None]} ' + \
                f'goal:{mpm.goal[None]}' + \
                f'finalpos:[{mpm.x[mpm.target_p[None]].x:.1f} {mpm.x[mpm.target_p[None]].y:.1f}]' + \
                f'reward:{mpm.total_reward[mpm.iter[None]]:.1f} ' + \
                f'finaldis:{np.sqrt(mpm.distance[None]):.1f} \n')
                # f'accuracy:{accuracy:.1f}%\n')
                fw.close()

            if mpm.iter[None] % POLICY_OUTPUT_SPAN == 0 :
                td3.save(policy_file+"/iter{}".format(mpm.iter[None]))
            if mpm.iter[None] % ACTUATION_OUTPUT_SPAN == 0 or mpm.iter[None] % OUTPUT_SPAN== 1 or mpm.iter[None] == NUM_ITERS-1:
                mpm.export_graph(actuation_file)
                # position.pop(0)
                
                # fig=plt.figure()
                # plt.title(f"position history, goal:{mpm.goal[None]}, reward:{mpm.total_reward[mpm.iter[None]]:.1f}")
                # plt.ylabel("position")
                # plt.xlabel("time")
                # plt.axhspan(0, 1, color = "olive", alpha=0.5)
                # plt.plot(position)
                # fig.savefig(actuation_file+f'/{mpm.iter[None]}position.png')
                # plt.close()

                fig=plt.figure()
                plt.title(f"xzposition, reward:{mpm.total_reward[mpm.iter[None]]:.1f}, final_distance:{np.sqrt(mpm.distance[None]):.1f},\n finalpos:[{mpm.x[mpm.target_p[None]].x:.1f} {mpm.x[mpm.target_p[None]].y:.1f}]\n")
                plt.ylabel("y")
                plt.xlabel("x")
                plt.plot(x_pos, z_pos, label="trajectory")
                plt.plot(x_pos[0], z_pos[0], color='blue', marker='>', label="start")
                plt.plot(x_pos[-1], z_pos[-1], color = 'purple', marker = '<',label="end")
                plt.plot(mpm.goal[None].x, mpm.goal[None].y, marker='*', label="goal")
                plt.grid()
                plt.legend(fontsize=15)
                plt.axis("equal")
                plt.minorticks_on()
                plt.grid(which="major", color="gray", linestyle="solid")
                plt.grid(which="minor", color="lightgray", linestyle="dotted")
                plt.tight_layout()
                fig.savefig(actuation_file+f'/{mpm.iter[None]}xzposition.png')
                plt.close()
                                
                # fig=plt.figure()
                # plt.title("v_x")
                # plt.ylabel("v_x")
                # plt.xlabel("t")
                # plt.plot(v_x)
                # # plt.plot(200, 5300, marker='*')
                # fig.savefig(actuation_file+f'/{mpm.iter[None]}vx.png')
                # plt.close()
                
                # fig=plt.figure()
                # plt.title("v_z")
                # plt.ylabel("v_z")
                # plt.xlabel("t")
                # plt.plot(v_z)
                # # plt.plot(200, 5300, marker='*')
                # fig.savefig(actuation_file+f'/{mpm.iter[None]}vz.png')
                # plt.close()
                
                fig = plt.figure()
                # los = mpm.total_reward.to_numpy()
                # ymax = los[0:LAST_ITER].max()
                # plt.title("reward history per span")
                plt.ylabel("Reward")
                plt.xlabel(f"time step [x{ACTUATION_SPAN}]")
                plt.plot(reward_per_span)
                plt.grid()
                plt.minorticks_on()
                plt.grid(which="major", color="gray", linestyle="solid")
                plt.grid(which="minor", color="lightgray", linestyle="dotted")                
                plt.tight_layout()
                # plt.xlim(0,LAST_ITER)
                # plt.ylim(0, ymax*1.05)
                fig.savefig(actuation_file+f'/{mpm.iter[None]}reward_per_span.png')
                plt.close()
                
                for i in range(max(mpm.iter[None]//2001, 0), 1+(mpm.iter[None]//2001)):
                    fig, ax = plt.subplots()
                    # los = mpm.total_reward.to_numpy()
                    # ymax = los[0:LAST_ITER].max()
                    # plt.title(f"reward history [iter:{2000*i} ~ {2000*(i+1)}]")
                    plt.ylabel("Reward")
                    plt.xlabel("Iterations")
                    # plt.axhspan(0, 5, color = "olive", alpha=0.5)
                    plt.plot(reward_history[2000*i:], label="reward", alpha = 0.4)
                    plt.plot(reward_moving_ave[2000*i:], color = 'purple', label="75 iter average")
                    plt.legend(fontsize=15)
                    # plt.xlim(0,LAST_ITER)
                    # plt.ylim(0, ymax*1.05)
                    plt.grid()
                    plt.minorticks_on()
                    plt.grid(which="major", color="gray", linestyle="solid")
                    plt.grid(which="minor", color="lightgray", linestyle="dotted")                
                    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    ax.ticklabel_format(style="sci",  axis="x",scilimits=(0,0))
                    plt.tight_layout()
                    fig.savefig(dr_original+f'/reward[iter:{2000*i} ~ {2000*(i+1)}].png')
                    plt.close()

                    fig, ax = plt.subplots()
                    # plt.title(f"final distance [iter:{2000*i} ~ {2000*(1+i)}]")
                    plt.ylabel("distance")
                    plt.xlabel("Iterations")
                    plt.plot(accuracy_history[2000*i:], label="final distance", alpha = 0.4)
                    plt.plot(dis_moving_ave[2000*i:], color = 'purple', label="75 iter average")
                    plt.grid()
                    plt.minorticks_on()
                    plt.grid(which="major", color="gray", linestyle="solid")
                    plt.grid(which="minor", color="lightgray", linestyle="dotted")                
                    plt.legend(fontsize=12)                
                    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    ax.ticklabel_format(style="sci",  axis="x",scilimits=(0,0))
                    plt.tight_layout()
                    fig.savefig(dr_original+f'/final_distance[iter:{2000*i} ~ {2000*(i+1)}].png')
                    plt.close()
                    
                fig, ax = plt.subplots()
                # los = mpm.total_reward.to_numpy()
                # ymax = los[0:LAST_ITER].max()
                plt.title(f"reward history, final avg: {reward_moving_ave[-1]:.1f}\n")
                plt.ylabel("Reward")
                plt.xlabel("Iterations")
                # plt.axhspan(0, 5, color = "olive", alpha=0.5)
                plt.plot(reward_history, label="reward", alpha =0.4)
                plt.plot(reward_moving_ave, color = 'purple', label="75 iter average")
                plt.legend(fontsize=15)
                # plt.xlim(0,LAST_ITER)
                # plt.ylim(0, ymax*1.05)
                plt.grid()
                plt.minorticks_on()
                plt.grid(which="major", color="gray", linestyle="solid")
                plt.grid(which="minor", color="lightgray", linestyle="dotted")                
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style="sci",  axis="x",scilimits=(0,0))
                plt.tight_layout()
                fig.savefig(dr_original+f'/overall_reward.png')
                plt.close()

                fig, ax = plt.subplots()
                plt.title(f"final distance, final avg: {dis_moving_ave[-1]:.1f}\n")
                plt.ylabel("distance")
                plt.xlabel("Iterations")
                plt.plot(accuracy_history, label="final distance", alpha =0.4)
                plt.plot(dis_moving_ave, color = 'purple', label="75 iter average")
                plt.grid()
                plt.minorticks_on()
                plt.grid(which="major", color="gray", linestyle="solid")
                plt.grid(which="minor", color="lightgray", linestyle="dotted")                
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style="sci",  axis="x",scilimits=(0,0))
                plt.legend(fontsize=15)
                plt.tight_layout()
                fig.savefig(dr_original+f'/overall_final_distance.png')
                plt.close()

                fig=plt.figure()
                # plt.title("critic loss")
                plt.ylabel("loss")
                plt.xlabel("steps")
                plt.yscale("log")
                plt.plot(td3.critic_loss1, label="critic loss 1")
                plt.plot(td3.critic_loss2, label="critic loss 2", color = "purple")
                plt.grid()
                plt.minorticks_on()
                plt.grid(which="major", color="gray", linestyle="solid")
                plt.grid(which="minor", color="lightgray", linestyle="dotted")                
                
                plt.legend(fontsize=15)
                plt.tight_layout()
                fig.savefig(dr_original+'/criticloss.png')
                plt.close()
                
                fig=plt.figure()
                # plt.title("actor loss")
                plt.ylabel("loss")
                plt.xlabel("steps")
                plt.plot(td3.actor_loss)
                plt.grid()
                plt.minorticks_on()
                plt.grid(which="major", color="gray", linestyle="solid")
                plt.grid(which="minor", color="lightgray", linestyle="dotted")                
                
                plt.tight_layout()
                fig.savefig(dr_original+'/actorloss.png')
                plt.close()
                
                # fig=plt.figure()
                # plt.title("total energy")
                # plt.ylabel("energy")
                # plt.xlabel("timestep")
                # plt.yscale("log")                
                # plt.plot(total_energy)
                # fig.savefig(actuation_file+f'/{mpm.iter[None]}total_energy.png')
                # plt.close()      

                # alpha_his.pop(0)
                # fig=plt.figure()
                # plt.title("aplha his")
                # plt.ylabel("alpha")
                # plt.xlabel("timestep")
                # plt.yscale("log")      
                # # plt.xscale("log")          
                # plt.plot(alpha_his)
                # fig.savefig(actuation_file+f'/{mpm.iter[None]}alpha.png')
                # plt.close()   
                
                fig = plt.figure(figsize=(12, 8))
                # plt.title("reward detail")
                plt.ylabel("reward")
                plt.xlabel("timestep")
                # plt.yscale("log")
                y = mpm.reward_detail.to_numpy().T
                for i, array in enumerate(y):
                    if i == 3:
                        plt.plot(array, label="total reward")
                    else:
                        plt.plot(array, label=f"reward #{i+1}")
                plt.grid()
                plt.minorticks_on()
                plt.grid(which="major", color="gray", linestyle="solid")
                plt.grid(which="minor", color="lightgray", linestyle="dotted")                
                plt.legend(fontsize=15 ,loc="center left", bbox_to_anchor=(1, 0.5))
                plt.tight_layout()
                plt.xlim(1, num_steps)
                fig.savefig(actuation_file+f'/{mpm.iter[None]}reward detail.png')
                plt.close()
                
            if  mpm.total_reward[mpm.iter[None]] > 1000 and mpm.distance[None] < 25.0:
                
        #     #random moving
        #     # goal_span = 5.0
        #     # goal_dif = rng.uniform(0, goal_span)
        #     # if mpm.goal[None].y < 4855.0 or mpm.goal[None].y > 5145.0:
        #     #     a *= -1
        #     # next_goal = mpm.goal[None].y + a * goal_dif
        #     # mpm.goal[None] = ti.Vector([0.0, 0.0, round(next_goal , 1)])
                mpm.export_graph(good_result_file)
                fig=plt.figure()
                plt.title(f"xzposition, reward:{mpm.total_reward[mpm.iter[None]]:.1f}, final_distance:{np.sqrt(mpm.distance[None]):.1f},\n finalpos:[{mpm.x[mpm.target_p[None]].x:.1f} {mpm.x[mpm.target_p[None]].y:.1f}]\n")
                plt.ylabel("y")
                plt.xlabel("x")
                plt.plot(x_pos, z_pos)
                plt.plot(x_pos[0], z_pos[0], color='blue', marker='>', label="start")
                plt.plot(x_pos[-1], z_pos[-1], color = 'purple', marker = '<', label="end")
                plt.plot(mpm.goal[None].x, mpm.goal[None].y, marker='*', label="goal")
                plt.grid()
                plt.minorticks_on()
                plt.grid(which="major", color="gray", linestyle="solid")
                plt.grid(which="minor", color="lightgray", linestyle="dotted")                
                
                plt.legend(fontsize=15)
                plt.axis("equal")                
                plt.tight_layout()
                fig.savefig(good_result_file+f'/{mpm.iter[None]}xzposition.png')
                plt.close()        
                

        #     #random
                
                # print(f"goal: {mpm.goal[None]}")
                # num_goal += 1
                # iter_per_goal_history.append(iter_per_goal)
                # iter_per_goal = 0
                # FIRST_TRY = True
                    # LAST_ITER = mpm.iter[None]
                    # mpm.export_csv(csv_file+f'/info_{mpm.iter[None]}.csv')


            if(math.isnan(mpm.distance[None])):
                print("NaN Error")
                reward_history.pop()
                accuracy_history.pop()
                mpm.export_graph(actuation_file)
                with open(dr_original+"/parameters.txt", mode="a") as fw:
                    fw.write(f'\niter:{mpm.iter[None]}\n' + f'{list(td3.actor.parameters())}')
                    fw.write(f'\n{list(td3.critic.parameters())}')
                    fw.close()   
                fig=plt.figure()
                # plt.title("critic loss")
                plt.ylabel("loss")
                plt.xlabel("steps")
                plt.yscale("log")
                plt.plot(td3.critic_loss1, label="critic loss 1")
                plt.plot(td3.critic_loss2, label="critic loss 2", color = "purple")
                plt.legend(fontsize=15)
                plt.grid()
                plt.minorticks_on()
                plt.grid(which="major", color="gray", linestyle="solid")
                plt.grid(which="minor", color="lightgray", linestyle="dotted")                
                
                plt.tight_layout()
                fig.savefig(dr_original+'/criticloss.png')
                plt.close()
                
                fig=plt.figure()
                # plt.title("actor loss")
                plt.ylabel("loss")
                plt.xlabel("steps")
                plt.plot(td3.actor_loss)
                plt.grid()
                plt.minorticks_on()
                plt.grid(which="major", color="gray", linestyle="solid")
                plt.grid(which="minor", color="lightgray", linestyle="dotted")                
                
                plt.tight_layout()
                fig.savefig(dr_original+'/actorloss.png')
                plt.close()
                
                fig=plt.figure()
                plt.title(f"xzposition, reward:{mpm.total_reward[mpm.iter[None]]:.1f}, final_distance:{np.sqrt(mpm.distance[None]):.1f},\n finalpos:[{mpm.x[mpm.target_p[None]].x:.1f} {mpm.x[mpm.target_p[None]].y:.1f}]\n")
                plt.ylabel("y")
                plt.xlabel("x")
                plt.plot(x_pos, z_pos, label="trajectory")
                plt.plot(x_pos[0], z_pos[0], color='blue', marker='>', label="start")
                plt.plot(x_pos[-1], z_pos[-1], color = 'purple', marker = '<', label="end")
                plt.plot(mpm.goal[None].x, mpm.goal[None].y, marker='*', label="goal")
                plt.grid()
                plt.minorticks_on()
                plt.grid(which="major", color="gray", linestyle="solid")
                plt.grid(which="minor", color="lightgray", linestyle="dotted")                
                
                plt.legend(fontsize=15)
                plt.axis("equal")               
                plt.tight_layout()
                fig.savefig(actuation_file+f'/{mpm.iter[None]}xzposition.png')
                plt.close()                                 
            #     print("error, loss:", mpm.reward_history[mpm.iter[None]])
            #     mpm.reset()
            #     dr = dr_original + '/iter{:03d}'.format(mpm.iter[None]-1)
            #     os.makedirs(dr, exist_ok=True)
            #     mpm.export_vtk(dr+'/mpm{:05d}.vtu'.format(0))
            
            #     while mpm.time_step[None] <= num_steps:
            #         # if 1000*(mpm.time_step[None])/num_steps%10 == 0:
            #         #     print(1000*(mpm.time_step[None])/num_steps)
                    
            #         mpm.grid_mass.fill(0)
            #         mpm.grid_momentum.fill(0)
            #         # if mpm.time_step[None] <= 0.5*num_steps:
            #         mpm.set_Fpxx()
            #         # else:
            #         #     mpm.Fp.fill(0)
                        
            #         with ti.Tape(mpm.total_energy):
            #             mpm.compute_total_energy()
            #             # mpm.compute_total_energy2()
            #         mpm.p2g()
            #         mpm.grid_op()
            #         mpm.g2p()
                    
            #         if mpm.time_step[None] % snap == 0:
            #             mpm.export_vtk(dr+'/mpm{:05d}.vtu'.format(mpm.time_step[None]))
            #         mpm.time_step[None] += 1
                break


            if mpm.iter[None] == WARMUP:
                iter_time_end = time.time()-iter_time_start
                estimated_total_time = int(iter_time_end * NUM_ITERS)
                h = estimated_total_time // 3600
                m = (estimated_total_time%3600)//60
                s = estimated_total_time%3600%60
                print("\nestimated time= {0:2d}h{1:2d}m{2:2d}s\n".format(h,m,s))

            if mpm.iter[None] != 0 and mpm.iter[None] % LR_DECAY_SPAN == 0:
                print("lr, noise decreased")
                expl_noise *= 0.7
                td3.policy_noise *= 0.7
                td3.noise_clip *= 0.7
                # mpm.reward_area[None] *= 0.9
                td3.scheduler1.step()
                td3.scheduler2.step()
                print("\nexpl_noise:", expl_noise, "policy noise:", td3.policy_noise, "noise clip:", td3.noise_clip, "actor lr:", td3.scheduler1.get_last_lr(), "critic lr:", td3.scheduler2.get_last_lr())
                
            # for y in range(190):
            #     td3.update()
          
            # iter_per_goal += 1
            mpm.iter[None] += 1
            # mpm.goal[None] = learning_goal[mpm.iter[None]%learning_goal_num]
            # R = rng.uniform(small_R, large_R)
            # THETA = rng.uniform(np.pi*0.25, np.pi*0.75)
            # mpm.goal[None] = ti.Vector([round(R * np.cos(THETA), 1), round(R * np.sin(THETA), 1)])
            mpm.goal[None]=ti.Vector([round(rng.uniform(-goal_length_x, goal_length_x) , 1), round(rng.uniform(goal_center_y-goal_length_y, goal_center_y+goal_length_y) , 1)])
            if mpm.iter[None] % OUTPUT_SPAN == 0 or mpm.iter[None] == NUM_ITERS-1:
                mpm.goal[None] = ti.Vector([-250.0, 5500.0])
            if mpm.iter[None] % OUTPUT_SPAN == 1 or mpm.iter[None] == NUM_ITERS:
                mpm.goal[None] = ti.Vector([250.0, 4500.0])

        with open(dr_original+"/parameters.txt", mode="a") as fw:
            fw.write('\nfinal parameters\n' + f'{list(td3.actor.parameters())}')
            fw.write(f'{list(td3.critic.parameters())}')
            fw.close()                
            

        print("finished")
        # actual_time = int(time.time()-TIME_START)
        # h = actual_time // 3600
        # m = (actual_time%3600)//60
        # s = actual_time%3600%60
        # print("total time= {0:2d}h{1:2d}m{2:2d}s\n".format(h,m,s))
        # print("number of goals:", num_goal)
        # print("average iter per goal:", mpm.iter[None]/num_goal)

        td3.save(policy_file+"/iter{}".format(mpm.iter[None]-1))
        


        # iter_per_goal_history.append(iter_per_goal)
        # fig=plt.figure()
        # plt.title("iter per goal")
        # plt.ylabel("number of iteration")
        # plt.xlabel("goal number")
        # plt.plot(iter_per_goal_history)
        # fig.savefig(dr_original+'/iter_per_goal.png')
        # plt.close()

        # fig=plt.figure()
        # plt.title("first loss per goal")
        # plt.ylabel("number of iteration")
        # plt.xlabel("goal number")
        # plt.plot(first_try)
        # fig.savefig(dr_original+'/first_loss_per_goal.png')
        # plt.close()

        # position.pop(0)
        # fig=plt.figure()
        # plt.title(f"position history, goal:{mpm.iter[None]-1}, reward:{reward_history[-1]:.2f}")
        # plt.ylabel("position")
        # plt.xlabel("time step")
        # plt.axhspan(-1, 1, color = "olive", alpha=0.5)
        # plt.plot(position)
        # fig.savefig(actuation_file+f'/position{mpm.iter[None]-1}.png')
        # plt.close()

        # fig=plt.figure()
        # actor_loss, critic_loss = td3.return_loss()
        # print("actor_loss:", actor_loss)
        # plt.title("loss of actor and critic")
        # plt.ylabel("number of learn")
        # plt.xlabel("loss")
        # plt.plot(actor_loss)
        # plt.plot(critic_loss)
        # fig.savefig(dr_original+'/actor-critic_loss.png')
        
        #agentのweightを出力できるが、多いので...で省略される
        # td3.get_actor()
    # color = plt.get_cmap("twilight")(np.linspace(0.1,0.45,test_num))
    color = plt.get_cmap("viridis")(np.linspace(0,0.9,test_num))
    if not (math.isnan(mpm.distance[None])):
        mpm.total_reward.fill(0)
        mpm.reward_detail.fill(0)
        mpm.reward_history.fill(0)
        print("start test")
        with open(dr_original+"/results.txt", mode="a") as fw:
            fw.write("\nstart test\n")
            fw.close()
        total_accuracy = 0
        total_reward = 0

        fig, axes   = plt.subplots(2, 5, figsize=(16,9), sharex="all", sharey="all")           
        for t in range(test_num):
            # mpm.total_reward[mpm.iter[None]] = 0
            mpm.goal[None] = goal[t]
            print(f"goal:{mpm.goal[None]}")
            with open(dr_original+"/results.txt", mode="a") as fw:
                fw.write(f"goal:{mpm.goal[None]}\n")
                fw.close()
            mpm.iter[None] = t+1
            mpm.reset()
            # position = []
            test_file = dr_original + '/test'
            os.makedirs(test_file, exist_ok=True)
            dr = test_file + f'/test_No.{t+1}'
            os.makedirs(dr, exist_ok=True)
            mpm.export_vtk(dr+'/test{0}_timestep{1:05d}.vtu'.format(t+1,0))
            mpm.export_goal(dr+f"/goal{t+1}")
            # csv_file = dr_original + '/csv'
            # os.makedirs(csv_file, exist_ok=True)
            actuation_file = test_file + '/test_actuation'
            os.makedirs(actuation_file, exist_ok=True)
            state = []
            action =[]
            x_pos = []
            z_pos = []
            while mpm.time_step[None] <= num_steps:
                
                for i in range(n_actuators):
                    mpm.actuation[mpm.time_step[None], i] = mpm.actuation[mpm.time_step[None]-1, i]

                if mpm.time_step[None] % ACTUATION_SPAN == 1:
                    state = []
                    for i in range(6):
                        state.append(mpm.state[mpm.time_step[None]-1][0,i])
                    # for i in range(6):
                    #     state.append(mpm.actuation[mpm.time_step[None]-ACTUATION_SPAN, i])
                    for i in range(20):
                        state.append((mpm.x[mpm.outer_state_p[i]].x+1000.0)/2000.0)
                        state.append((mpm.x[mpm.outer_state_p[i]].y)/7000.0)
                    # for i in range(dim):
                    state.append((mpm.goal[None][0]+1000.0)/2000.0)
                    state.append((mpm.goal[None][1]-3000.0)/4000.0)
                    # state = 10.0*state
                    action = td3.select_action(np.array(state))
                # state.append(mpm.actuation[mpm.time_step[None]-1])
                # state.append(mpm.time_step[None]/ num_steps)
                # action = td3.select_action(np.array(state))
                    for i in range(n_actuators):
                        mpm.actuation[mpm.time_step[None], i] = action[i]#*(1-mpm.time_step[None]/num_steps)
                # print(mpm.actuation[mpm.time_step[None]])
                mpm.step()
                x_pos.append(mpm.x[mpm.target_p[None]].x)
                z_pos.append(mpm.x[mpm.target_p[None]].y)
                
                _reward = gauss(abs(mpm.pre_distance[None] - ti.sqrt(mpm.distance[None])))
                mpm.reward_detail[mpm.time_step[None], 0] = _reward  
                reward = _reward
                
                # _reward = 2.0*gauss(abs(mpm.x[mpm.target_p[None]].x-mpm.goal[None].x))
                # mpm.reward_detail[mpm.time_step[None], 1] = _reward  
                # reward += _reward
                
                _reward = -0.1*sum([i**2 for i in action])
                mpm.reward_detail[mpm.time_step[None], 1] = _reward  
                reward += _reward
                
                # _reward = 3.0*(mpm.pre_distance[None] - ti.sqrt(mpm.distance[None]))
                # mpm.pre_distance[None] = ti.sqrt(mpm.distance[None])
                # mpm.reward_detail[mpm.time_step[None], 3] = _reward
                # reward += _reward
                
                _reward = -mpm.total_energy[None]*1e-7
                mpm.reward_detail[mpm.time_step[None], 2] = _reward
                reward += _reward
                
                # _reward = 0.0
                # for i in range(n_actuators):
                #     _reward += -2.0*np.power(mpm.actuation[mpm.time_step[None], i]-mpm.actuation[mpm.time_step[None]-1, i], 2)
                # mpm.reward_detail[mpm.time_step[None], 5] = _reward  
                # reward += _reward

                mpm.reward_detail[mpm.time_step[None], 3] = reward  
                
                mpm.reward_history[mpm.iter[None], mpm.time_step[None]] = reward
                mpm.total_reward[mpm.iter[None]] += reward
                
                
                if mpm.time_step[None] % snap == 0:
                    mpm.export_vtk(dr+'/test{0}_timestep{1:05d}.vtu'.format(t+1,mpm.time_step[None]))
                    
                        
                distance = np.sqrt(mpm.distance[None])
                # position.append(distance)
                mpm.time_step[None] += 1
                
            mpm.export_graph(actuation_file)
            # position.pop(0)
            
            # print(distance)
            accuracy = distance
            total_accuracy += accuracy
            total_reward += mpm.total_reward[mpm.iter[None]]
            plt.suptitle(f"xzposition, avg reward:{total_reward/float(test_num):.1f}, avg distance:{total_accuracy/float(test_num):.1f}")
            # plt.ylabel("y")
            # plt.xlabel("x")
            axes[0][4-t%5].plot(x_pos, z_pos, color = color[t], alpha =0.5)
            # plt.plot(x_pos[0], z_pos[0], color='blue', marker='>')
            axes[0][4-t%5].plot(x_pos[-1], z_pos[-1], color = color[t], marker = '<')
            axes[0][4-t%5].plot(mpm.goal[None].x, mpm.goal[None].y, marker='*', color = color[t])
            # axes[0][4-t%5].axis("equal")               
            axes[0][4-t%5].grid(True)
            axes[0][4-t%5].minorticks_on()
            axes[0][4-t%5].grid(which="major", color="gray", linestyle="solid")
            axes[0][4-t%5].grid(which="minor", color="lightgray", linestyle="dotted")                
            axes[0][4-t%5].set(xlabel='x', ylabel='y')  # 全てのサブプロットに対してラベルを設定
            axes[0][4-t%5].label_outer()             
            
            axes[1][t//5].plot(x_pos, z_pos, color = color[t], alpha =0.5)
            # plt.plot(x_pos[0], z_pos[0], color='blue', marker='>')
            axes[1][t//5].plot(x_pos[-1], z_pos[-1], color = color[t], marker = '<')
            axes[1][t//5].plot(mpm.goal[None].x, mpm.goal[None].y, marker='*', color = color[t])
            # axes[0][t//5].axis("equal")               
            axes[1][t//5].grid(True)
            axes[1][t//5].minorticks_on()
            axes[1][t//5].grid(which="major", color="gray", linestyle="solid")
            axes[1][t//5].grid(which="minor", color="lightgray", linestyle="dotted")                
            
            axes[1][t//5].set(xlabel='x', ylabel='y')  # 全てのサブプロットに対してラベルを設定
            axes[1][t//5].label_outer()             
            fig.tight_layout()
            fig.savefig(actuation_file+'/test_position.png')          
            print(f'  test:{t+1}\n'\
                , f'  reward:{mpm.total_reward[mpm.iter[None]]:.1f}\n'\
                , f'  finalpos:[{mpm.x[mpm.target_p[None]].x:.1f} {mpm.x[mpm.target_p[None]].y:.1f}]\n'\
                , f'  final distance:{distance:.1f}\n' )
                # , f'accuracy:{accuracy:.1f}%')
            with open(dr_original+"/results.txt", mode="a") as fw:
                fw.write(f'  test:{t+1}\n' + \
                f'  reward:{mpm.total_reward[mpm.iter[None]]:.1f}\n' + \
                f'  finalpos:[{mpm.x[mpm.target_p[None]].x:.1f} {mpm.x[mpm.target_p[None]].y:.1f}]\n' +\
                f'  final distance:{distance:.1f} \n')
                # f'accuracy:{accuracy:.1f}%\n')
                fw.close()
            mpm.iter[None] += 1
        plt.close()
            
        print(f"average reward:{total_reward/float(test_num):.2f},average final distance:{total_accuracy/float(test_num):.1f}")
        with open(dr_original+"/results.txt", mode="a") as fw:
            fw.write(f"average reward:{total_reward/float(test_num):.2f}, average final distance:{total_accuracy/float(test_num):.1f}")
            fw.close()
    # if TEST_ONLY:
    #     with open(dr_original+"/parameters.txt", mode="a") as fw:
    #         fw.write('tested parameter\n' + f'{list(td3.actor.parameters())}')
    #         fw.close()    

if __name__ == '__main__':
    main()
