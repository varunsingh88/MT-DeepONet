"""
Author: Varun Kumar
Date: Jan 3 2025
Purpose: script to save data and errors for Darcy problem
Contact: varun_kumar2@brown.edu
"""

import numpy as np


def save_data(f_test, u_pred, u_test, save_results_to, domain, exp_name):
    domain = domain
    save_results_to = save_results_to

    err_u = np.mean(np.linalg.norm(u_pred - u_test, 2, axis=1) / np.linalg.norm(u_test, 2, axis=1))

    if domain == 'source':
        print('Relative L2 Error (Source): %.3f' % (err_u))
        err_u = np.reshape(err_u, (-1, 1))
        np.savetxt(save_results_to + '/err', err_u, fmt='%e')

    else:
        print('Relative L2 Error (Target): %.3f' % (err_u))
        err_u = np.reshape(err_u, (-1, 1))
        np.savetxt(save_results_to + '/err', err_u, fmt='%e')
