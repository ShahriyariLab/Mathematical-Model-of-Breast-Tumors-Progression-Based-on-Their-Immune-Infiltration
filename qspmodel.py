# from odesolver import *
import numpy as np
import scipy.optimize as op
from itertools import permutations
from scipy.integrate import odeint
import pandas as pd


def Sensitivity_function(x,t,QSP):
    nparam=QSP.qspcore.nparam
    nvar=QSP.qspcore.nvar
    x=x.reshape(nparam+1,nvar)
    dxdt=np.empty(x.shape)
    dxdt[0]=QSP(x[0],t)
    dxdt[1:]=np.dot(QSP.Ju(x[0],t),x[1:].T).T+ QSP.Jp(x[0],t)
    return dxdt.flatten()


# Object class that defines the functions for the appropriate QSP Model
class Breast_QSP_Functions(object):
    def __init__(self,SSrestrictions=np.ones(38)):
        self.nparam=75
        self.nvar=17
        self.SSscale = SSrestrictions
        self.variable_names=['Naive T-cells', 'helper T-cells', 'cytotoxic cells', 'Treg-cells', 'Naive Dendritic cells', 'Dendritic cells', 'Naive Macrophages', 'Macrophages',
                        'Cancer cells', 'Necrotic cells','Adipocytes', 'HMGB1','IL-12','IL-10','Estrogen', 'IFN-gamma', 'IL-6']
        self.parameter_names=['lambda_{T_hH}','lamda_{T_hD}','lambda_{T_hIL_{12}}','lambda_{T_hE}',
                         'lambda_{T_cE}','lambda_{T_cD}','lambda_{T_cIL_{12}}',
                         'lambda_{T_rD}','lambda_{T_rE}',
                         'lambda_{DC}','lambda_{DH}','lambda_{DE}',
                         'lambda_{MIL_{10}}','lambda_{MI_gamma}','lambda_{MIL_{12}}','lambda_{MT_h}','lambda_{ME}',
                         'lambda_{C}','lambda_{CIL_6}','lambda_{CA}',
                         'lambda_{A}',
                         'lambda_{HD}','lambda_{HN}','lambda_{HM}','lambda_{HT_c}','lambda_{HC}',
                         'lambda_{IL_{12}M}','lambda_{IL_{12}D}','lambda_{IL_{12}T_h}','lambda_{IL_{12}T_c}',
                         'lambda_{IL_{10}M}','lambda_{IL_{10}D}','lambda_{IL_{10}T_r}','lambda_{IL_{10}T_h}','lambda_{IL_{10}T_c}','lambda_{IL_{10}C}',
                         'lambda_{EA}','lambda_{E}',
                         'lambda_{I_gammaT_c}','lambda_{I_gammaT_h}','lambda_{I_gammaD}',
                         'lambda_{IL_6A}','lambda_{IL_6M}','lambda_{IL_6D}',
                         'delta_{T_hT_r}','delta_{T_hIL_{10}}','delta_{T_h}',
                         'delta_{T_cIL_{10}}','delta_{T_CT_r}','delta_{T_c}',
                         'delta_{T_r}',
                         'delta_{T_N}',
                         'delta_{DC}','delta_{D}',
                         'delta_{D_N}',
                         'delta_{M}',
                         'delta_{M_N}',
                         'delta_{CT_c}','delta_{CI_gamma}','delta_{C}',
                         'delta_{A}',
                         'delta_{N}',
                         'delta_{H}',
                         'delta_{IL_{12}}',
                         'delta_{IL_{10}}',
                         'delta_{E}',
                         'delta_{I_gamma}',
                         'delta_{IL_6}',
                         'A_{T_N}','A_{D_N}','A_{M}',
                         'alpha_{NC}','C_0','A_0','E_0']

    def __call__(self,x,t,p):
        # ODE right-hand side
        return np.array([p[68]-(p[0]*x[11]+p[1]*x[5]+p[2]*x[12]+p[3]*x[14])*x[0]-(p[4]*x[14]+p[5]*x[5]+p[6]*x[12])*x[0]-(p[7]*x[5]+p[8]*x[14]+p[51])*x[0],\
                        (p[0]*x[11]+p[1]*x[5]+p[2]*x[12]+p[3]*x[14])*x[0]-(p[44]*x[3]+p[45]*x[13]+p[46])*x[1],\
                        (p[4]*x[14]+p[5]*x[5]+p[6]*x[12])*x[0]-(p[48]*x[3]+p[47]*x[13]+p[49])*x[2],\
                        (p[7]*x[5]+p[8]*x[14])*x[0]-p[50]*x[3],\
                        p[69]-(p[9]*x[8]+p[10]*x[11]+p[11]*x[14])*x[4]-p[54]*x[4],\
                        (p[9]*x[8]+p[10]*x[11]+p[11]*x[14])*x[4]-(p[52]*x[8]+p[53])*x[5],\
                        p[70]-(p[12]*x[13]+p[13]*x[15]+p[14]*x[12]+p[15]*x[1]+p[16]*x[14]+p[56])*x[6],\
                        (p[12]*x[13]+p[13]*x[15]+p[14]*x[12]+p[15]*x[1]+p[16]*x[14])*x[6]-p[55]*x[7],\
                        (p[17]+p[18]*x[16]+p[19]*x[10])*(1-x[8]/p[72])*x[8]-(p[57]*x[2]+p[58]*x[15]+p[59])*x[8],\
                        p[71]*(p[58]*x[15]+p[57]*x[2]+p[59])*x[8]-p[61]*x[9],\
                        (p[20]*x[10])*(1-x[10]/p[73])-p[60]*x[10],\
                        p[21]*x[5]+p[22]*x[9]+p[23]*x[7]+p[24]*x[2]+p[25]*x[8]-p[62]*x[11],\
                        p[26]*x[7]+p[27]*x[5]+p[28]*x[1]+p[29]*x[2]-p[63]*x[12],\
                        p[30]*x[7]+p[31]*x[5]+p[32]*x[3]+p[33]*x[1]+p[34]*x[2]+p[35]*x[8]-p[64]*x[13],\
                        p[36]*x[10]+p[37]*x[14]*(1-x[14]/p[74])-p[65]*x[14],\
                        p[38]*x[2]+p[39]*x[1]+p[40]*x[14]*x[5]-p[66]*x[15],\
                        p[41]*x[10]+p[42]*x[7]+p[43]*x[5]-p[67]*x[16]])
    def Ju(self,x,t,p):

        # Jacobian with respect to variables
        return np.array([[-p[51] - p[0]*x[11] - p[2]*x[12] - p[6]*x[12] - p[3]*x[14] - p[4]*x[14] - p[8]*x[14] - p[1]*x[5] - p[5]*x[5] - p[7]*x[5], 0, 0, 0, 0, -p[1]*x[0] - p[5]*x[0] - p[7]*x[0], 0, 0, 0, 0, 0, -p[0]*x[0], -p[2]*x[0] - p[6]*x[0], 0, -p[3]*x[0] - p[4]*x[0] - p[8]*x[0], 0, 0],\
                        [p[0]*x[11] + p[2]*x[12] + p[3]*x[14] + p[1]*x[5], -p[46] - p[45]*x[13] - p[44]*x[3], 0, -p[44]*x[1], 0, p[1]*x[0], 0, 0, 0, 0, 0, p[0]*x[0], p[2]*x[0], -p[45]*x[1], p[3]*x[0], 0, 0],\
                        [p[6]*x[12] + p[4]*x[14] + p[5]*x[5], 0, -p[49] - p[47]*x[13] - p[48]*x[3], -p[48]*x[2], 0, p[5]*x[0], 0, 0, 0, 0, 0, 0, p[6]*x[0], -p[47]*x[2], p[4]*x[0], 0, 0],\
                        [p[8]*x[14] + p[7]*x[5], 0, 0, -p[50], 0, p[7]*x[0], 0, 0, 0, 0, 0, 0, 0, 0, p[8]*x[0], 0, 0],\
                        [0, 0, 0, 0, -p[54] - p[10]*x[11] - p[11]*x[14] -  p[9]*x[8], 0, 0, 0, -p[9]*x[4], 0, 0, -p[10]*x[4], 0, 0, -p[11]*x[4], 0, 0],\
                        [0, 0, 0, 0, p[10]*x[11] + p[11]*x[14] + p[9]*x[8], -p[53] - p[52]*x[8], 0, 0, p[9]*x[4] - p[52]*x[5], 0, 0, p[10]*x[4], 0, 0, p[11]*x[4], 0, 0],\
                        [0, -p[15]*x[6], 0, 0, 0, 0, -p[56] - p[15]*x[1] - p[14]*x[12] - p[12]*x[13] - p[16]*x[14] - p[13]*x[15], 0, 0, 0, 0, 0, -p[14]*x[6], -p[12]*x[6], -p[16]*x[6], -p[13]*x[6], 0],\
                        [0, p[15]*x[6], 0, 0, 0, 0, p[15]*x[1] + p[14]*x[12] + p[12]*x[13] + p[16]*x[14] + p[13]*x[15], -p[55], 0, 0, 0, 0, p[14]*x[6], p[12]*x[6], p[16]*x[6], p[13]*x[6], 0],\
                        [0, 0, -p[57]*x[8], 0, 0, 0, 0, 0, -p[59] - p[58]*x[15] - p[57]*x[2] - ((p[17] + p[19]*x[10] + p[18]*x[16])*x[8])/p[72] + (p[17] + p[19]*x[10] + p[18]*x[16])*(1 - x[8]/p[72]), 0, p[19]*x[8]*(1 - x[8]/p[72]), 0, 0, 0, 0, -p[58]*x[8], p[18]*x[8]*(1 - x[8]/p[72])],\
                        [0, 0, p[57]*p[71]*x[8], 0, 0, 0, 0, 0, p[71]*(p[59] + p[58]*x[15] + p[57]*x[2]), -p[61], 0, 0, 0, 0, 0, p[58]*p[71]*x[8], 0],\
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -p[60] - (p[20]*x[10])/p[73] + p[20]*(1 - x[10]/p[73]), 0, 0, 0, 0, 0, 0],\
                        [0, 0, p[24], 0, 0, p[21], 0, p[23], p[25], p[22], 0, -p[62], 0, 0, 0, 0, 0],\
                        [0, p[28], p[29], 0, 0, p[27], 0, p[26], 0, 0, 0, 0, -p[63], 0, 0, 0, 0],\
                        [0, p[33], p[34], p[32], 0, p[31], 0, p[30], p[35], 0, 0, 0, 0, -p[64], 0, 0, 0],\
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, p[36], 0, 0, 0, -p[65] - (p[37]*x[14])/p[74] + p[37]*(1 - x[14]/p[74]), 0, 0],\
                        [0, p[39], p[38], 0, 0, p[40]*x[14], 0, 0, 0, 0, 0, 0, 0, 0, p[40]*x[5], -p[66], 0],\
                        [0, 0, 0, 0, 0, p[43], 0, p[42], 0, 0, p[41], 0, 0, 0, 0, 0, -p[67]]])
    def Jp(self,x,t,p):
        # Jacobian with respect to the parameters
        # p[1], p[6], p[8], p[9], p[15], p[19], p[20], p[21], p[27], p[31], p[36], p[39], p[42], p[61], p[68], p[69], p[70]
        return np.array([[-x[0]*x[11], x[0]*x[11], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[5], x[0]*x[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[12], x[0]*x[12], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[14], x[0]*x[14], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[14], 0, x[0]*x[14], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[5], 0, x[0]*x[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[12], 0, x[0]*x[12], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[5], 0, 0, x[0]*x[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [-x[0]*x[14], 0, 0, x[0]*x[14], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, -x[4]*x[8], x[4]*x[8], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, -x[11]*x[4], x[11]*x[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, -x[14]*x[4], x[14]*x[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, -x[13]*x[6], x[13]*x[6], 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, -x[15]*x[6], x[15]*x[6], 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, -x[12]*x[6], x[12]*x[6], 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, -x[1]*x[6], x[1]*x[6], 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, -x[14]*x[6], x[14]*x[6], 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, x[8]*(1 - x[8]/p[72]), 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, x[16]*x[8]*(1 - x[8]/p[72]), 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, x[10]*x[8]*(1 - x[8]/p[72]), 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[10]*(1 - x[10]/p[73]), 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[5], 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[9], 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[7], 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[2], 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[8], 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[7], 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[5], 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[2], 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[7], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[5], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[3], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[2], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[8], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[10], 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[14]*(1 - x[14]/p[74]), 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[2], 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[14]*x[5], 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[10]],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[7]],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[5]],\
                      [0, -x[1]*x[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, -x[1]*x[13], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, -x[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, -x[13]*x[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, -x[2]*x[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, -x[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, -x[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [-x[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, -x[5]*x[8], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, -x[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, -x[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, -x[7], 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, -x[6], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, -x[2]*x[8], p[71]*x[2]*x[8], 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, -x[15]*x[8], p[71]*x[15]*x[8], 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, -x[8], p[71]*x[8], 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[10], 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -x[9], 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[11], 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[12], 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[13], 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[14], 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[15], 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[16]],\
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, (p[59] + p[58]*x[15] + p[57]*x[2])*x[8], 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, ((p[17] + p[19]*x[10] +  p[18]*x[16])*(x[8]*x[8]))/(p[72]*p[72]), 0, 0, 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (p[20]*(x[10]*x[10]))/(p[73]*p[73]), 0, 0, 0, 0, 0, 0],\
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (p[37]*(x[14]*x[14]))/(p[74]*p[74]), 0, 0]])
    def SS_system(self,p,frac):
        # compute the system and restrictions with non-dimensional steady states at 1
        # pre-defined rates, extreme and global mean values are hardcoded here
        # cell fractions are given as [Tn Th Tc Tr Dn D M M0 C N]
        #in case of non-dimensional use 1's
        x=np.ones(self.nvar);
        #in case of dimensional use:
        #x = frac
        # rates acquired from bio research [p[46],p[49],p[50],p[51],p[53],p[55],p[60],p[62],p[63],p[64],p[65],p[66],p[67]]
        globvals=np.array([0.231,0.406,0.231,0.000949,0.277,0.0198,0.0028,18,128,4.62,4.16,33.3,1.07])
        # maximal values of each variable across all patients [\"Tn\", \"Th\", \"Tc\", \"Tr\", \"Dn\", \"D\", \"Mn\", \"M\", \"C\", \"N\", \"A\", \"H\", \"IL12\", \"IL10\", \"E\", \"Ig\", \"IL6\"]
        extremevals=np.array([42704.74471,15987.51107,24949.82652,11735.53076,6506.50228,8273.739237,45767.4528,52470.57289,425754.4018,101936.0142,261167.2797,11.59553491,15.15933397,5.962509503,18.68070021,8.768141948,10.73423998])
        # average values of each variable across all patients [Tc mu1 Ig Gb]
        meanvals=np.array([4402.978812,3729.677968,4129.095508,1163.230463,389.4736518,262.750754,8521.345255,12961.49494,46375.88899,6309.182087,71120.0947,5.993292007,7.155235085,3.505263906,10.32542172,3.675417979,3.906352659])



        return np.array([p[68]-(p[0]*x[11]+p[1]*x[5]+p[2]*x[12]+p[3]*x[14])*x[0]-(p[4]*x[14]+p[5]*x[5]+p[6]*x[12])*x[0]-(p[7]*x[5]+p[8]*x[14]+p[51])*x[0],\
                (p[0]*x[11]+p[1]*x[5]+p[2]*x[12]+p[3]*x[14])*x[0]-(p[44]*x[3]+p[45]*x[13]+p[46])*x[1],\
                        (p[4]*x[14]+p[5]*x[5]+p[6]*x[12])*x[0]-(p[48]*x[3]+p[47]*x[13]+p[49])*x[2],\
                        (p[7]*x[5]+p[8]*x[14])*x[0]-p[50]*x[3],\
                        p[69]-(p[9]*x[8]+p[10]*x[11]+p[11]*x[14])*x[4]-p[54]*x[4],\
                        (p[9]*x[8]+p[10]*x[11]+p[11]*x[14])*x[4]-(p[52]*x[8]+p[53])*x[5],\
                        p[70]-(p[12]*x[13]+p[13]*x[15]+p[14]*x[12]+p[15]*x[1]+p[16]*x[14]+p[56])*x[6],\
                        (p[12]*x[13]+p[13]*x[15]+p[14]*x[12]+p[15]*x[1]+p[16]*x[14])*x[6]-p[55]*x[7],\
                        (p[17]+p[18]*x[16]+p[19]*x[10])*(1-x[8]/p[72])*x[8]-(p[57]*x[2]+p[58]*x[15]+p[59])*x[8],\
                        p[71]*(p[58]*x[15]+p[57]*x[2]+p[59])*x[8]-p[61]*x[9],\
                        (p[20]*x[10])*(1-x[10]/p[73])-p[60]*x[10],\
                        p[21]*x[5]+p[22]*x[9]+p[23]*x[7]+p[24]*x[2]+p[25]*x[8]-p[62]*x[11],\
                        p[26]*x[7]+p[27]*x[5]+p[28]*x[1]+p[29]*x[2]-p[63]*x[12],\
                        p[30]*x[7]+p[31]*x[5]+p[32]*x[3]+p[33]*x[1]+p[34]*x[2]+p[35]*x[8]-p[64]*x[13],\
                        p[36]*x[10]+p[37]*x[14]*(1-x[14]/p[74])-p[65]*x[14],\
                        p[38]*x[2]+p[39]*x[1]+p[40]*x[14]*x[5]-p[66]*x[15],\
                        p[41]*x[10]+p[42]*x[7]+p[43]*x[5]-p[67]*x[16],\
                        (p[17] + p[18]*(meanvals[16]/frac[16]) + p[19]*(meanvals[10]/frac[10])) - p[59] - 0.013329753,\
                        p[17] - (p[57]*(meanvals[2]/frac[2]) + p[58]*(meanvals[15]/frac[15]) + p[59]) - 0.00207529096,\
                        self.SSscale[0]*p[0]-p[1]*(extremevals[5]/frac[5])/(200*(extremevals[11]/frac[11])),\
                        self.SSscale[1]*p[2] - p[1]*(extremevals[5]/frac[5])/(200*(extremevals[12]/frac[12])),\
                        self.SSscale[2]*p[3] - p[1]*(extremevals[5]/frac[5])/(200*(extremevals[14]/frac[14])),\
                        self.SSscale[3]*p[45] - p[44]*(extremevals[3]/frac[3])/(extremevals[13]/frac[13]),\
                        self.SSscale[4]*p[44] - p[46]*20/(extremevals[3]/frac[3]),\
                        self.SSscale[5]*p[5] - 0.01*p[4]*(extremevals[14]/frac[14])/(extremevals[5]/frac[5]),\
                        self.SSscale[6]*p[6] - 0.005*p[4]*(extremevals[14]/frac[14])/(extremevals[12]/frac[12]),\
                        self.SSscale[7]*p[47] - p[49]*20/(extremevals[13]/frac[13]),\
                        self.SSscale[8]*p[48] - p[49]*20/(extremevals[3]/frac[3]),\
                        self.SSscale[9]*p[8] - p[7]*(extremevals[5]/frac[5])/(4*extremevals[14]/frac[14]),\
                        self.SSscale[10]*p[10] - 2*p[9]*(extremevals[8]/frac[8])/(extremevals[11]/frac[11]),\
                        self.SSscale[11]*p[11] - 2*p[9]*(extremevals[8]/frac[8])/(extremevals[14]/frac[14]),\
                        self.SSscale[12]*p[52] - 0.277/(extremevals[8]/frac[8]),\
                        self.SSscale[13]*p[13] - 10*p[12]*(extremevals[13]/frac[13])/(10*extremevals[15]/frac[15]),\
                        self.SSscale[14]*p[14] - 10*p[12]*(extremevals[13]/frac[13])/(10*extremevals[12]/frac[12]),\
                        self.SSscale[15]*p[15] - 10*p[12]*(extremevals[13]/frac[13])/(extremevals[1]/frac[1]),\
                        self.SSscale[16]*p[16] - 10*p[12]*(extremevals[13]/frac[13])/(10*extremevals[14]/frac[14]),\
                        self.SSscale[17]*p[22] - p[21]*(extremevals[5]/frac[5])/(extremevals[9]/frac[9]),\
                        self.SSscale[18]*p[23] - p[21]*(extremevals[5]/frac[5])/(10*extremevals[7]/frac[7]),\
                        self.SSscale[19]*p[24] - p[21]*(extremevals[5]/frac[5])/(extremevals[2]/frac[2]),\
                        self.SSscale[20]*p[25] - p[21]*(extremevals[5]/frac[5])/(extremevals[8]/frac[8]),\
                        self.SSscale[21]*p[27] - p[26]*(extremevals[7]/frac[7])/(extremevals[5]/frac[5]),\
                        self.SSscale[22]*p[28] - p[26]*(extremevals[7]/frac[7])/(extremevals[1]/frac[1]),\
                        self.SSscale[23]*p[29] - p[26]*(extremevals[7]/frac[7])/(extremevals[2]/frac[2]),\
                        self.SSscale[24]*p[31] - p[30]*(extremevals[7]/frac[7])/(extremevals[5]/frac[5]),\
                        self.SSscale[25]*p[32] - p[30]*(extremevals[7]/frac[7])/(extremevals[3]/frac[3]),\
                        self.SSscale[26]*p[33] - p[30]*(extremevals[7]/frac[7])/(extremevals[1]/frac[1]),\
                        self.SSscale[27]*p[34] - p[30]*(extremevals[7]/frac[7])/(extremevals[2]/frac[2]),\
                        self.SSscale[28]*p[35] - p[30]*(extremevals[7]/frac[7])/(extremevals[8]/frac[8]),\
                        self.SSscale[29]*p[39] - p[38]*(extremevals[2]/frac[2])/(5*extremevals[1]/frac[1]),\
                        self.SSscale[30]*p[40] - p[38]*(extremevals[2]/frac[2])/(5*extremevals[5]/frac[5]),\
                        self.SSscale[31]*p[43] - p[41]*(extremevals[10]/frac[10])/(extremevals[5]/frac[5]),\
                        self.SSscale[32]*p[42] - p[41]*(extremevals[10]/frac[10])/(extremevals[7]/frac[7]),\
                        self.SSscale[33]*p[71] - 0.5*(frac[8]/frac[9]),\
                        self.SSscale[34]*p[36] - p[37]/(20*extremevals[10]/frac[10]),\
                        self.SSscale[35]*p[19] - 0.5*(p[18]*(extremevals[16]/frac[16]))/(extremevals[10]/frac[10]),\
                        self.SSscale[36]*p[59] - 6*(p[58]*(extremevals[15]/frac[15])),\
                        self.SSscale[37]*p[57] - 6*p[58]*(extremevals[15]/frac[15])/(extremevals[2]/frac[2]),\
                        p[72] - 2*(extremevals[8]/frac[8]),\
                        p[73] - (extremevals[10]/frac[10]),\
                        p[74] - 1.5*(extremevals[14]/frac[14]),\
                        p[54] - p[53],\
                        p[56] - p[55],\
                        p[46] - globvals[0],\
                        p[49] - globvals[1],\
                        p[50] - globvals[2],\
                        p[51] - globvals[3],\
                        p[53] - globvals[4],\
                        p[55] - globvals[5],\
                        p[60] - globvals[6],\
                        p[62] - globvals[7],\
                        p[63] - globvals[8],\
                        p[64] - globvals[9],\
                        p[65] - globvals[10],\
                        p[66] - globvals[11],\
                        p[67] - globvals[12]])


class QSP:
    def __init__(self,parameters,qspcore=Breast_QSP_Functions()):
        self.qspcore=qspcore
        self.p=parameters;
    def set_parameters(self,parameters):
        self.p=parameters;
    def steady_state(self):
        # compute steady state with current parameters
        IC=np.ones(self.qspcore.nvar);
        return op.fsolve((lambda x: self.qspcore(x,0,self.p)),IC,fprime=(lambda x: self.qspcore.Ju(x,0,self.p)),xtol=1e-7,maxfev=10000)  #This might need to change
    def Sensitivity(self,method='steady',t=None,IC=None,params=None,variables=None):

        if method=='time':
            if IC is None:
                raise Exception('Error: Need initial conditions for time integration. Set IC=')
                return None
            if t is None:
                raise Exception('Error: Need time values for time integration. Set t=')
                return None

            nparam=self.qspcore.nparam
            nvar=self.qspcore.nvar
            initial=np.zeros((nparam+1,nvar));
            initial[0]=IC
            return np.mean(odeint(Sensitivity_function, initial.flatten(), t, args=(self, )) ,axis=0).reshape(nparam+1,nvar)[1:]
        elif method=='split':
            if not hasattr(self,'variable_par'):
                raise Exception('error: parameter splitting is not set. use "initiate_parameter_split" method')
                return None
            if params is None:
                raise Exception('error: Need parameter values for split sensitivity. Set params=')
                return None
            elif len(params)!=sum(self.variable_par):  #what is variable_par?
                raise Exception('error: wrong number of parameters given')
                return None

            if IC is None:
                IC=np.ones(self.qspcore.nvar);
            par=np.copy(self.p)
            par[self.variable_par]=np.copy(params)

            u=op.fsolve((lambda x: self.qspcore(x,0,par)),IC,fprime=(lambda x: self.qspcore.Ju(x,0,par)),xtol=1e-7,maxfev=10000)
            if variables is None:
                return -np.dot(self.qspcore.Jp(u,0,self.p),np.linalg.inv(self.qspcore.Ju(u,0,self.p).T))[self.variable_par]
            else:
                return -np.dot(self.qspcore.Jp(u,0,self.p),np.linalg.inv(self.qspcore.Ju(u,0,self.p).T))[self.variable_par,variables]
        else:
            u=self.steady_state()
            return -np.dot(self.qspcore.Jp(u,0,self.p),np.linalg.inv(self.qspcore.Ju(u,0,self.p).T))
    def __call__(self,x,t):
        return self.qspcore(x,t,self.p)
    def Ju(self,x,t):
        return self.qspcore.Ju(x,t,self.p)
    def Jp(self,x,t):
        return self.qspcore.Jp(x,t,self.p)
    def variable_names(self):return self.qspcore.variable_names
    def parameter_names(self):return self.qspcore.parameter_names
    def solve_ode(self, t, IC, method='default'):
        # Solve ode system with either default 1e4 time steps or given time discretization
        # t - time: for 'default' needs start and end time
        #           for 'given' needs full array of time discretization points
        # IC - initial conditions
        # method: 'default' - given interval divided by 10000 time steps
        #         'given' - given time discretization
        if method=='given':
            return odeint((lambda x,t: self.qspcore(x,t,self.p)), IC, t,
                            Dfun=(lambda x,t: self.qspcore.Ju(x,t,self.p))), t
        else:
            return odeint((lambda x,t: self.qspcore(x,t,self.p)), IC, np.linspace(min(t), max(t), 10001),
                            Dfun=(lambda x,t: self.qspcore.Ju(x,t,self.p))), np.linspace(min(t), max(t), 10001)

    def initiate_parameter_split(self,variable_par):
        # splits the parameters into fixed and variable for further fittin
        # variable_par - boolean array same size as parameter array indicating which parameters are variable
        if (variable_par.dtype!='bool') or (len(variable_par)!=self.qspcore.nparam):
            raise Exception('error: wrong parameter indicator')
            return None
        self.variable_par=np.copy(variable_par)

    def solve_ode_split(self, t, IC, params):
        # Solve ode system with adjusted variable parameters
        #   using either default 1e4 time steps or given time discretization
        # t - time: needs full array of time discretization points
        # IC - initial conditions
        # params - parameters to update for this solution
        if not hasattr(self,'variable_par'):
            raise Exception('error: parameter splitting is not set. use "initiate_parameter_split" method')
            return None
        if len(params)!=sum(self.variable_par):
            raise Exception('error: wrong number of parameters given')
            return None
        par=np.copy(self.p)
        par[self.variable_par]=np.copy(params)
        return odeint((lambda x,t: self.qspcore(x,t,par)), IC, t,
                            Dfun=(lambda x,t: self.qspcore.Ju(x,t,par)))

    @classmethod
    def from_cell_data(class_object, fracs, qspcore=Breast_QSP_Functions()):
        params=op.fsolve((lambda p,fracs: qspcore.SS_system(p,fracs)),np.ones(qspcore.nparam),
                         args=(fracs,))
        return class_object(params)
