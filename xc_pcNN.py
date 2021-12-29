# -*- coding: utf-8 -*-
#!/usr/bin/env python
# %%
import numpy as np
import random
import math
from pyscf import dft
import jax
from jax import numpy as jnp
from jax import grad, jit, vmap, value_and_grad
import torch
import torch.nn as nn
import os

ATANSIGMA = 1.0  # decaying speed of arctan
ATANSIGMA2 = 1.0
SCALE = 1  # 2.0
SCALE_C = 1  # 2
DELTA_X = 1  # delta to kill higher order effect in    polynomial
DELTA_C = 1


THIRD = 1.0/3
THIRD2 = 2.0/3
THIRD4 = 4.0/3
THIRD5 = 5.0/3

seed = 0
random.seed(seed)
np.random.seed(100)
# np.random.seed()
PI = math.pi
PIH = PI*0.5
IPIH = 1.0/PIH

LObound = 1.174
LObound_point = 2.0*math.atanh(2.0/LObound-1.0)

tpoint = math.tan(math.pi*(1.0/LObound-0.5))
bpoint = 1.0  # LObound/PI/(1+tpoint**2)
LObound_point2 = tpoint*bpoint

# %%

# %%


class Net:
    def __init__(self):
        super(Net, self).__init__()  # モジュール継承, load_state_dictなどに必要
        self.hidden = 100
        self.tansigma = ATANSIGMA
        self.forward_j = jit(self.forward)
        self.vforward = jit(vmap(self.forward_j, in_axes=(0, None)))
        self.forward_cj = jit(self.forward_c)
        self.vforward_c = jit(vmap(self.forward_cj, in_axes=(0, None)))
        self.calc_s0_j = self.calc_s0
        self.product_j = self.product
        self.connect_j = jit(
            vmap(value_and_grad(self.connect, argnums=0), in_axes=(0, None)))
        self.connect_cj = jit(
            vmap(value_and_grad(self.connect_c, argnums=0), in_axes=(0, None)))
        self.con = jit(vmap(self.product_j, in_axes=(0, None, 0)))
        self.makeg_j = jit(self.makeg)
        self.gen_conds_j = jit(self.gen_conds)
        self.gen_conds_cj = jit(self.gen_conds_c)
        # self.mkmat()

    def mkmat(self, seed=0, scale=SCALE, scale_c=SCALE_C):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.fc1 = nn.Linear(2, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.fc4 = nn.Linear(self.hidden, 1)
        self.w1 = jnp.array(self.fc1.weight.data.numpy(),
                            dtype=np.float32)*scale
        self.w2 = jnp.array(self.fc2.weight.data.numpy(),
                            dtype=np.float32)*scale
        self.w3 = jnp.array(self.fc3.weight.data.numpy(),
                            dtype=np.float32)*scale
        self.w4 = jnp.array(self.fc4.weight.data.numpy(),
                            dtype=np.float32)*scale
        self.b1 = jnp.array(self.fc1.bias.data.numpy(), dtype=np.float32)*scale
        self.b2 = jnp.array(self.fc2.bias.data.numpy(), dtype=np.float32)*scale
        self.b3 = jnp.array(self.fc3.bias.data.numpy(), dtype=np.float32)*scale
        self.b4 = jnp.array(self.fc4.bias.data.numpy(), dtype=np.float32)*scale
        self.fc1c = nn.Linear(2, self.hidden)
        self.fc2c = nn.Linear(self.hidden, self.hidden)
        self.fc3c = nn.Linear(self.hidden*2, self.hidden)
        self.fc4c = nn.Linear(self.hidden, 1)
        self.w1c = jnp.array(self.fc1c.weight.data.numpy(),
                             dtype=np.float32)*scale_c
        self.w2c = jnp.array(self.fc2c.weight.data.numpy(),
                             dtype=np.float32)*scale_c
        self.w3c = jnp.array(self.fc3c.weight.data.numpy(),
                             dtype=np.float32)*scale_c
        self.w4c = jnp.array(self.fc4c.weight.data.numpy(),
                             dtype=np.float32)*scale_c
        self.b1c = jnp.array(self.fc1c.bias.data.numpy(),
                             dtype=np.float32)*scale_c
        self.b2c = jnp.array(self.fc2c.bias.data.numpy(),
                             dtype=np.float32)*scale_c
        self.b3c = jnp.array(self.fc3c.bias.data.numpy(),
                             dtype=np.float32)*scale_c
        self.b4c = jnp.array(self.fc4c.bias.data.numpy(),
                             dtype=np.float32)*scale_c

    def save(self, fname):
        os.system("mkdir "+fname)
        np.save(fname+"/w1", np.array(self.w1, dtype=np.float32))
        np.save(fname+"/w2", np.array(self.w2, dtype=np.float32))
        np.save(fname+"/w3", np.array(self.w3, dtype=np.float32))
        np.save(fname+"/w4", np.array(self.w4, dtype=np.float32))
        np.save(fname+"/w1c", np.array(self.w1c, dtype=np.float32))
        np.save(fname+"/w2c", np.array(self.w2c, dtype=np.float32))
        np.save(fname+"/w3c", np.array(self.w3c, dtype=np.float32))
        np.save(fname+"/w4c", np.array(self.w4c, dtype=np.float32))
        np.save(fname+"/b1", np.array(self.b1, dtype=np.float32))
        np.save(fname+"/b2", np.array(self.b2, dtype=np.float32))
        np.save(fname+"/b3", np.array(self.b3, dtype=np.float32))
        np.save(fname+"/b4", np.array(self.b4, dtype=np.float32))
        np.save(fname+"/b1c", np.array(self.b1c, dtype=np.float32))
        np.save(fname+"/b2c", np.array(self.b2c, dtype=np.float32))
        np.save(fname+"/b3c", np.array(self.b3c, dtype=np.float32))
        np.save(fname+"/b4c", np.array(self.b4c, dtype=np.float32))

    def load(self, fname):
        # debug = 1.0
        self.w1 = jnp.array(np.load(fname+"/w1.npy"),
                            dtype=np.float32)  # *debug
        self.w2 = jnp.array(np.load(fname+"/w2.npy"),
                            dtype=np.float32)  # *debug
        self.w3 = jnp.array(np.load(fname+"/w3.npy"),
                            dtype=np.float32)  # *debug
        self.w4 = jnp.array(np.load(fname+"/w4.npy"),
                            dtype=np.float32)  # *debug
        self.w1c = jnp.array(np.load(fname+"/w1c.npy"),
                             dtype=np.float32)  # *debug
        self.w2c = jnp.array(np.load(fname+"/w2c.npy"),
                             dtype=np.float32)  # *debug
        self.w3c = jnp.array(np.load(fname+"/w3c.npy"),
                             dtype=np.float32)  # *debug
        self.w4c = jnp.array(np.load(fname+"/w4c.npy"),
                             dtype=np.float32)  # *debug
        self.b1 = jnp.array(np.load(fname+"/b1.npy"),
                            dtype=np.float32)  # *debug
        self.b2 = jnp.array(np.load(fname+"/b2.npy"),
                            dtype=np.float32)  # *debug
        self.b3 = jnp.array(np.load(fname+"/b3.npy"),
                            dtype=np.float32)  # *debug
        self.b4 = jnp.array(np.load(fname+"/b4.npy"),
                            dtype=np.float32)  # *debug
        self.b1c = jnp.array(np.load(fname+"/b1c.npy"),
                             dtype=np.float32)  # *debug
        self.b2c = jnp.array(np.load(fname+"/b2c.npy"),
                             dtype=np.float32)  # *debug
        self.b3c = jnp.array(np.load(fname+"/b3c.npy"),
                             dtype=np.float32)  # *debug
        self.b4c = jnp.array(np.load(fname+"/b4c.npy"),
                             dtype=np.float32)  # *debug

    def makeg(self, x):
        unif = (x[1]+x[0]+1e-7)**THIRD

        t0 = unif
        div = 1.0/(x[1]+x[0]+1e-7)
        t1 = ((1+(x[0]-x[1])*div)**THIRD4+(1-(x[0]-x[1])*div)**THIRD4)*0.5
        t2 = ((x[2]+x[4]+2*x[3]+10**(-56/3))**0.5)/unif**4
        tauunif = 0.3*(3*PI**2)**THIRD2*unif**5
        t3 = (x[5]+x[6])/tauunif-1.0

        t = jnp.tanh(jnp.stack((t0, t1, t2, t3), axis=-1)/self.tansigma)
        return t

    def shifted_softplus0(self, x):
        l2 = 0.6931471805599453
        tmp = jnp.exp(2.0*l2*(x))
        f = 1.0/l2*jnp.log(1+tmp)
        return f

    def shifted_softplus1(self, x):
        l2 = math.log(2)
        tmp = np.exp(2.0*l2*(x-1.0))
        f = 1.0/l2*np.log(1+tmp)
        df = 2.0*tmp/(1+tmp)
        return f, df

    def forward(self, t, params):
        w1, w2, w3, w4, b1, b2, b3, b4 = params

        g1 = self.shifted_softplus0(jnp.matmul(t, w1.T)+b1)
        g2 = self.shifted_softplus0(jnp.matmul(g1, w2.T)+b2)
        g3 = self.shifted_softplus0(jnp.matmul(g2, w3.T)+b3)
        g4 = jnp.matmul(g3, w4.T)+b4
        return g4[0]

    def forward_c(self, t, params):  # t1,t2の通すべきNNが逆だったので訓練やり直し
        params_x = params[:8]
        params_c = params[8:]
        w1, w2, w3, w4, b1, b2, b3, b4 = params_x
        w1c, w2c, w3c, w4c, b1c, b2c, b3c, b4c = params_c
        t1 = t[0:2]
        t2 = t[2:4]

        g1 = self.shifted_softplus0(jnp.matmul(t1, w1c.T)+b1c)
        g2 = self.shifted_softplus0(jnp.matmul(g1, w2c.T)+b2c)

        g1_x = self.shifted_softplus0(jnp.matmul(t2, w1.T)+b1)
        g2_x = self.shifted_softplus0(jnp.matmul(g1_x, w2.T)+b2)

        g2_c = jnp.concatenate((g2, g2_x))
        g3 = self.shifted_softplus0(jnp.matmul(g2_c, w3c.T)+b3c)
        g4 = jnp.matmul(g3, w4c.T)+b4c
        return g4[0]

    def gen_conds(self, g):
        rho, zeta, s, tau = g

        g0s = jnp.array([[0, 0], [1, tau]])
        self.ncon_x = g0s.shape[0]
        c_s_inf = 1.0
        f0s = jnp.array([1.0, c_s_inf])  # ,c_s_mid
        return g0s, f0s

    # vectorize index: (0,None,0) index->index[i], dis_ij->dis_ij[i]
    def product(self, index, dis, dis_ij):
        rdis = jnp.roll(dis, -index)
        rdis_ij = jnp.roll(dis_ij, -index)
        sigma = 0

        denomi = jnp.prod(jnp.where(rdis_ij < 1e-7, 1.0, rdis)[1:])
        numer = jnp.prod(jnp.where(rdis_ij < 1e-7, 1.0, rdis_ij)[1:])
        return denomi/numer

    def connect(self, n, params):
        g = self.makeg_j(n)
        gx = g[2:]
        g0x, f0x = self.gen_conds_j(g)
        inds = jnp.arange(0, self.ncon_x)
        f_nn = self.forward(gx, params)
        f_g0 = self.vforward(g0x, params)
        delta = DELTA_X

        dis = jnp.sum((gx-g0x)**2, axis=1)  # descriptor distance
        dis = jnp.tanh(dis/delta**2)  # dimentionless
        g0x = g0x.reshape((self.ncon_x, 1, -1))
        g0xt = g0x.transpose((1, 0, 2))
        dis_ij = jnp.sum((g0x-g0xt)**2, axis=2)

        dis_ij = jnp.tanh(dis_ij/delta**2)  # dimentionless
        cs = self.con(inds.reshape(-1, 1), dis, dis_ij)

        self.gx = gx
        self.cs = cs
        fs = (f_nn-f_g0)+f0x
        total = jnp.dot(fs, cs)/jnp.sum(cs)
        return total

    def gen_conds_c(self, g, params):
        rho, zeta, s, tau = g
        g0s = jnp.array(((rho, zeta, 0, 0),
                         (0, zeta, s, tau), (1.0, zeta, s, tau)))
        self.ncon_c = g0s.shape[0]
        c_rho_inf = 1.0
        g_lowdens = jnp.array((rho, 0, s, tau))
        g_lowdens0 = jnp.array((0, 0, s, tau))

        f_nn_lowdens = self.forward_cj(
            g_lowdens, params)-self.forward_cj(g_lowdens0, params)+1.0  # To make fNN→1 at NNparameters=0

        f0s = jnp.array((1.0, f_nn_lowdens, c_rho_inf))  # ,c_s_mid
        return g0s, f0s

    def connect_c(self, n, params):
        gc = self.makeg_j(n)
        g0c, f0c = self.gen_conds_cj(gc, params)
        inds = jnp.arange(0, self.ncon_c)
        f_nn = self.forward_cj(gc, params)
        f_g0 = self.vforward_c(g0c, params)
        delta = DELTA_C

        dis = jnp.sum((gc-g0c)**2, axis=1)  # descriptor distance
        dis = jnp.tanh(dis/delta**2)

        g0c = g0c.reshape((self.ncon_c, 1, -1))
        g0ct = g0c.transpose((1, 0, 2))
        dis_ij = jnp.sum((g0c-g0ct)**2, axis=2)
        dis_ij = jnp.tanh(dis_ij/delta**2)

        cs = self.con(inds.reshape(-1, 1), dis, dis_ij)

        self.gc = gc
        self.cs = cs
        fs = (f_nn-f_g0)+f0c  # (f_nn-f_g0)+f0c
        total = jnp.dot(fs, cs)/jnp.sum(cs)
        return total
        # return f_nn

    def calc_s0(self, rho):
        rho0, dx, dy, dz, lapl, tau = rho[:6]
        gamma1 = gamma2 = gamma12 = (dx**2+dy**2+dz**2)*0.25
        rho01 = rho02 = rho0*0.5
        tau1 = tau2 = tau*0.5
        n = jnp.stack((rho01, rho02, gamma1, gamma12,
                       gamma2, tau1, tau2), axis=-1)
        N = n.shape[0]
        params = (self.w1, self.w2, self.w3, self.w4,
                  self.b1, self.b2, self.b3, self.b4)
        fx, gr = self.connect_j(n, params)
        return fx, gr

    def calc_c(self, rho, spin):
        if spin != 0:
            rho1 = rho[0]
            rho2 = rho[1]
            rho01, dx1, dy1, dz1, lapl1, tau1 = rho1[:6]
            rho02, dx2, dy2, dz2, lapl2, tau2 = rho2[:6]
            rho0 = rho01+rho02
            gamma1 = dx1**2+dy1**2+dz1**2
            gamma2 = dx2**2+dy2**2+dz2**2
            gamma12 = dx1*dx2+dy1*dy2+dz1*dz2
        else:
            rho0, dx, dy, dz, lapl, tau = rho[:6]
            gamma1 = gamma2 = gamma12 = (dx**2+dy**2+dz**2)*0.25
            rho01 = rho02 = rho0*0.5
            tau1 = tau2 = tau*0.5

        n = np.stack((rho01, rho02, gamma1, gamma12,
                      gamma2, tau1, tau2), axis=-1)

        params_c = (self.w1, self.w2, self.w3, self.w4, self.b1, self.b2, self.b3, self.b4,
                    self.w1c, self.w2c, self.w3c, self.w4c, self.b1c, self.b2c, self.b3c, self.b4c)
        # print(len(params_c))
        fc, gr = self.connect_cj(n, params_c)
        return fc, gr

    def eval_x(self, n, spin):
        if spin == 0:
            N = n[0].shape[0]
            rho0 = n[0]
            fx, gr = map(np.array, self.calc_s0(n))
            fx, df = self.shifted_softplus1(fx)
            gr *= df.reshape((-1, 1))
            escan, vscan = dft.xcfun.eval_xc('scan,', n, spin=0)[0:2]
            ex = fx*escan
            vlapl = np.zeros(N)
            vrho = vscan[0]*fx+rho0*escan*(gr[:, 0]+gr[:, 1])/2
            vgamma = vscan[1]*fx+rho0*escan*(gr[:, 2]+gr[:, 4]+gr[:, 3])/4
            vtau = vscan[3]*fx+rho0*escan*(gr[:, 5]+gr[:, 6])/2

        else:
            n1, n2 = n
            rho01 = n1[0]
            rho02 = n2[0]
            N = rho01.shape[0]
            fx1, gr1 = map(np.array, self.calc_s0(n1*2))
            fx1, df = self.shifted_softplus1(fx1)
            gr1 *= df.reshape((-1, 1))
            fx2, gr2 = map(np.array, self.calc_s0(n2*2))
            fx2, df2 = self.shifted_softplus1(fx2)
            gr2 *= df2.reshape((-1, 1))

            escan1, vscan1 = dft.xcfun.eval_xc('scan,', n1*2, spin=0)[0:2]
            escan2, vscan2 = dft.xcfun.eval_xc('scan,', n2*2, spin=0)[0:2]
            # print(fx1)
            ex = (rho01*escan1*fx1+rho02*escan2*fx2)/(rho01+rho02)

            vrho1 = vscan1[0]*fx1+2*rho01*escan1*gr1[:, 0]
            vgamma1 = vscan1[1]*fx1*2+4*rho01*escan1*gr1[:, 2]
            vtau1 = vscan1[3]*fx1+2*rho01*escan1*gr1[:, 5]

            vrho2 = vscan2[0]*fx2+2*rho02*escan2*gr2[:, 0]
            vgamma2 = vscan2[1]*fx2*2+4*rho02*escan2*gr2[:, 2]
            vtau2 = vscan2[3]*fx2+2*rho02*escan2*gr2[:, 5]

            vrho = np.stack((vrho1, vrho2), -1)
            vgamma = np.stack((vgamma1, np.zeros(N), vgamma2), -1)
            vlapl = np.zeros((N, 2))
            vtau = np.stack((vtau1, vtau2), -1)

        vx = (vrho, vgamma, vlapl, vtau)
        return ex, vx

    def eval_c(self, n, spin):
        fc, gr = map(np.array, self.calc_c(n, spin))
        fc, df = self.shifted_softplus1(fc)
        gr *= df.reshape((-1, 1))

        escan, vscan = dft.xcfun.eval_xc(',scan', n, spin)[0:2]
        ec = fc*escan
        N = escan.shape[0]
        if spin != 0:
            n1, n2 = n
            rho01 = n1[0]
            rho02 = n2[0]
            rho0 = rho01+rho02
            vlapl = np.zeros((N, 2))
            vrho = np.stack(((vscan[0][:, 0]*fc+rho0*escan*gr[:, 0]),
                             (vscan[0][:, 1]*fc+rho0*escan*gr[:, 1])), axis=-1)
            vgamma = np.stack(((vscan[1][:, 0]*fc+rho0*escan*gr[:, 2]), (vscan[1][:, 1]
                                                                         * fc+rho0*escan*gr[:, 3]), (vscan[1][:, 2]*fc+rho0*escan*gr[:, 4])), axis=-1)
            vtau = np.stack(((vscan[3][:, 0]*fc+rho0*escan*gr[:, 5]),
                             (vscan[3][:, 1]*fc+rho0*escan*gr[:, 6])), axis=-1)
        else:
            rho0 = n[0]
            vlapl = np.zeros(N)
            vrho = vscan[0]*fc+rho0*escan*(gr[:, 0]+gr[:, 1])/2
            vgamma = vscan[1]*fc+rho0*escan*(gr[:, 2]+gr[:, 4]+gr[:, 3])/4
            vtau = vscan[3]*fc+rho0*escan*(gr[:, 5]+gr[:, 6])/2
        vc = (vrho, vgamma, vlapl, vtau)

        return ec, vc

    def eval_xc(self, xc_code, rho, spin, relativity=0, deriv=2, verbose=None):
        ex, vx = self.eval_x(rho, spin)
        ec, vc = self.eval_c(rho, spin)

        exc = ex+ec
        vxc = tuple(vx[i]+vc[i] for i in range(4))

        return exc, vxc, None, None
