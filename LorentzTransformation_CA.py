# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 08:08:46 2023
Lorenzt Transformation - Special Relativity

@author: chunauntan
"""

import numpy as np
import matplotlib.pyplot as plt

C_VACUUM = 299792458 #m/sec

def x_ly2m(x_ly):
    x_m = x_ly * C_VACUUM
    return x_m

def gamma(u, c = 1):
    beta= u/c
    return 1.0 / np.sqrt(1- np.dot(beta,beta))

def LT_rt2rptp(r =[0,0,0], t = 0, u =[0,0,0]):
    # the transformed coordinate system x'y'z't' is moving to the in the positive direction ofstationary XYZ coordinate system
    # refer to Feynman Vol1 Lect17 eq. 17.1
    u = np.array(u)
    beta = u
    gamma = 1.0 / np.sqrt(1- np.dot(beta,beta))
    rp = (r - u*t)*gamma
    tp = (t - np.dot(u,r))*gamma
    print('r = ', r, '  and t = ', t,' -->')
    print('rp = ', np.round(rp, 4), '  and tp = ', np.round(tp,4),'\n')
    return rp, tp

def LT_xt_xtp(x = 0, t= 0, uxp = 0, c =1):
    """
    Lorenzt transformation from x and t to x' and t'.
    Assuming uxp is the velocity of x'-t' inertial frame of reference.
    To change to tranformation of x'-t' to x-t, only convert uxp to -uxp.

    Parameters
    ----------
    x : TYPE, optional
        coordinate x in x-t. The default is 0.
    t : TYPE, optional
        coordinate t (time) in x-t. The default is 0.
    uxp : TYPE, optional
        speed of the moving frame x'-t' to the right of x axis. The default is 0.
    c : TYPE, optional
        speed of light. The default is 1.

    Returns
    -------
    x : TYPE
        coordinate x of x-t frame
    t : TYPE
        coordinate t of x-t frame
    xp : TYPE
        coordinate x' of x'-t' frame
    tp : TYPE
        coordinate t' of x'-t' frame

    """
    # eq. # 0
    # print('eq.#0')
    g = gamma(uxp, c)
    xp = (x - uxp*t) * g
    tp = ( t - (uxp * x / c**2) ) * g
    return x ,t, xp, tp

def x_xp2t_tp(x, xp, uxp, c =1):
    """
    # transform from (x, ?) and (x', ?) to  (x, t) and (x', t')

    """
    # eq #1
    # print('eq.#1')
    g = gamma(uxp, c)
    t = (x - xp/g ) / uxp
    tp = LT_xt_xtp(x, t, uxp, c)[3]
    return x, t, xp, tp

def t_xp2x_tp(t, xp, uxp, c=1):
    """
    # transform from (?, t) and (x', ?) to  (x, t) and (x', t')

    """
    # eq #2
    # print('eq.#2')
    g = gamma(uxp, c)
    x = xp / g + uxp*t
    tp = LT_xt_xtp(x, t, uxp, c)[3]
    return x, t, xp, tp

def x_tp2xp_t(x, tp, uxp, c=1):
    """
    # transform from (x, ?) and (?, t') to  (x, t) and (x', t')

    """
    # eq #3
    # print('eq.#3')
    g = gamma(uxp, c)
    t = (uxp * x /c**2) + tp / g
    xp = LT_xt_xtp(x, t, uxp, c)[2]
    return x, t, xp, tp

def t_tp2x_xp(t, tp, uxp, c=1):
    """
    # transform from (?, t) and (?, t') to  (x, t) and (x', t')

    """
    # eq #4
    # print('eq.#4')
    g = gamma(uxp, c)
    x = (t - tp/g)*c**2/uxp
    xp = LT_xt_xtp(x, t, uxp, c)[2]
    return x, t, xp, tp

def LT_xt(XT = ['',''], XT_p = ['',''], uxp =0.01, c=1):
    """
    Lorenzt Transformation within X-T to X'-T' inertial frame of reference
    as long as 2 of the 4 params is available, this function will generate
    the other 2.

    Parameters
    ----------
    XT : list, optional
        XT[0] = x
        XT[1] = t
        '' means unknown
        The default is ['',''].
    XT_p : list, optional
       XT_p[0] = x'
       XT_p[1] = t'
       '' means unknown.
       The default is ['',''].
    uxp : float, optional
        speed of X'-T' along x axis. The default is 0.01.
    c : float, optional
        Speed of light. The default is 1.

    Returns
    -------
    list
        [x,t] is coordinate of x-t frame
    list
        [xp,tp] is coordinate of x'-t' frame

    """
    if type(XT_p[0])==str and type(XT_p[1])==str:
        # case given x and t
        # print('CASE#0')
        x = XT[0]
        t = XT[1]
        x, t, xp, tp = LT_xt_xtp(x, t, uxp, c)
    elif type(XT[0])==str and type(XT[1])==str:
        # case given x' and t'
        # print('CASE#1')
        xp = XT_p[0]
        tp = XT_p[1]
        x, t, xp, tp = LT_xt_xtp(xp, tp, -uxp, c)
    elif type(XT[1]) == str and type(XT_p[1])==str:
        # case#2  given x and x'
        # print('CASE#2')
        x = XT[0]
        xp = XT_p[0]
        x, t, xp, tp = x_xp2t_tp(x, xp, uxp, c =1)
    elif type(XT[0]) == str and type(XT_p[1])==str:
        # case#3  given t and x'
        # print('CASE#3')
        t = XT[1]
        xp = XT_p[0]
        x, t, xp, tp = t_xp2x_tp(t, xp, uxp, c=1)
    elif type(XT[1]) == str and type(XT_p[0])==str:
        # case#4  given x and t'
        # print('CASE#4')
        x = XT[0]
        tp = XT_p[1]
        x, t, xp, tp = x_tp2xp_t(x, tp, uxp, c=1)
    elif type(XT[1]) == str and type(XT_p[0])==str:
        # case#5  given t and t'
        # print('CASE#5')
        t = XT[1]
        tp = XT_p[1]
        x, t, xp, tp = t_tp2x_xp(t, tp, uxp, c=1)
    print('[x, t] = [', np.round(x,4) ,', ',np.round(t,4), '] <-->')
    print('[xp, tp] = [', np.round(xp,4) ,', ',np.round(tp,4), ']','\n')
    return [x, t], [xp, tp]

# =============================================================================
# MAIN PROGRAM !!!!
# =============================================================================
if __name__ == "__main__":
    # choose program to execute
    programN = 1
    if programN == 1:
        c = 1 # set speed of light constant
        u = 0.8
        xt = [8, '']
        xt_p =[0 ,'']
        XT, XT_p = LT_xt(XT= xt, XT_p = xt_p, uxp = u)
