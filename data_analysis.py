import random
from scipy.stats import sem
import statistics
import math

########################################################################
#Input: time and population states series, n - replicon copy number, N - max no. of cells, dl - detection limit
#Time and population series (t_ts and pop_ts) are converted to af_ts - allele frequency series. 
#Then different types of estimations are retrieved from these data
########################################################################


def detectablehet(t_ts, pop_ts, n, N, dl=0.1, contribution_dic=None):
    '''
    Calculates AF for each population state, then goes through the array once to estimate the fraction of time when the heteroplasmy was detectable.
    The function also returns af_ts for later calculations.
    contribution_dic (parameter sequencing scheme) has cell_names as keys and contribution proportion as values.
    '''
    detectable = 0
    total_time = 0
    af_ts = []
    for i in range(len(pop_ts)):
        ti = pop_ts[i].state
        N_curr = len(ti)
        if contribution_dic != None:
            try:
                mut_curr = sum([x[0]*contribution_dic[x[1]] for x in ti])
            except TypeError:
                mut_curr = sum([sum(x[0])*contribution_dic[x[1]] for x in ti])
            rep_total = sum([n*contribution_dic[x[1]] for x in ti])
        else:
            try:
                mut_curr = sum([x[0] for x in ti])
            except TypeError:
                mut_curr = sum([sum(x[0]) for x in ti])
            rep_total = N_curr * n
        af_curr = mut_curr / rep_total
        af_ts.append(af_curr)
        if dl <= af_curr <= (1 - dl):
            detectable += t_ts[i]
        total_time += t_ts[i]
    
    return af_ts, detectable / total_time

      

def ageFixedLevels(t_ts, pop_ts, nrep, ncp):
    '''
    Calculate time until the variant is fixed on the level of a)plastids, b)cells for the fist time
    '''
    cp_fixed_time = []
    cell_fixed_time = []
    #the number is the time, the boolian is whether the time should be increased (as the fixation wasn't reached yet)
    curr_cp_fixed_time, curr_cell_fixed_time = (0, True), (0, True)
    for i in range(len(pop_ts)):
        ti = pop_ts[i].state
        if pop_ts[i].het:
            for cell in ti:
                #fixed on the level of cp
                if nrep in [x for x in cell[0]]:
                    if curr_cp_fixed_time[1]:
                        curr_cp_fixed_time = (curr_cp_fixed_time[0] + 1, False)
                #fixed on the level of a cell
                if sum(cell[0]) == nrep * ncp:
                    if curr_cell_fixed_time[1]:
                        curr_cell_fixed_time = (curr_cell_fixed_time[0] + 1, False)
            if curr_cp_fixed_time[1]:
                curr_cp_fixed_time = (curr_cp_fixed_time[0] + t_ts[i], True)
            if curr_cell_fixed_time[1]:
                curr_cell_fixed_time = (curr_cell_fixed_time[0] + t_ts[i], True)
        else:
            cp_fixed_time.append(curr_cp_fixed_time[0])
            cell_fixed_time.append(curr_cell_fixed_time[0])
            curr_cp_fixed_time, curr_cell_fixed_time = (0, True), (0, True)
    #add from the last mutation
    cp_fixed_time.append(curr_cp_fixed_time[0])
    cell_fixed_time.append(curr_cell_fixed_time[0])
    
    #return statistics.mean(cp_fixed_time), sem(cp_fixed_time), statistics.mean(cell_fixed_time), sem(cell_fixed_time)
    return cp_fixed_time, cell_fixed_time
    


def det_nondet_het(t_ts, pop_ts, nrep, ncp, N, mutfix, dl=0.05, contribution_dic=None):
    '''
    Calculate contribution of the intermutation time and non-detectable mutations.
    This is the main function used to analysed data from the simulation.
    '''
    n = nrep * ncp
    af_ts, prob = detectablehet(t_ts, pop_ts, n, N, dl=dl, contribution_dic=contribution_dic)
    cp_fixed_time, cell_fixed_time = ageFixedLevels(t_ts, pop_ts, nrep, ncp)
    #to evaluate the effect on homo and hetero stages separately
    homo_time, het_det_time, het_nondet_time = 0, 0, 0
    het_det_list, het_age_dist = [], {}
    fixed_mut = 0
    #to count undetected mutations at any point
    curr_mut_det, detected_mut = False, 0 
    curr_mut_det_time, curr_seg_time, curr_homo_time = 0, 0, 0
    for i in range(len(af_ts)):
        if af_ts[i] == 1:
            fixed_mut += 1
            het_det_list.append((curr_mut_det_time, curr_seg_time, curr_homo_time, 'fixed'))
            curr_mut_det_time, curr_seg_time = 0, 0
            curr_mut_det = False
            detected_mut += 1
            homo_time += t_ts[i]
            curr_homo_time = t_ts[i]
        elif af_ts[i] == 0:
            if curr_seg_time > 0:
                het_det_list.append((curr_mut_det_time, curr_seg_time, curr_homo_time, 'lost'))
                curr_mut_det_time, curr_seg_time = 0, 0
                if curr_mut_det:
                    detected_mut += 1
                curr_mut_det = False
                curr_homo_time = t_ts[i]
            else:
                curr_homo_time += t_ts[i]
            homo_time += t_ts[i]
        elif dl <= af_ts[i] <= (1 - dl):
            het_det_time += t_ts[i]
            curr_mut_det = True
            curr_mut_det_time += t_ts[i]
            curr_seg_time += t_ts[i]
            for gen in range(t_ts[i]):
                het_age_dist.setdefault(curr_seg_time - gen, 0)
                het_age_dist[curr_seg_time - gen] += 1
        else:
            het_nondet_time += t_ts[i]
            curr_seg_time += t_ts[i]
    
    
    frac_hetero = (het_det_time + het_nondet_time) / (homo_time + het_det_time + het_nondet_time)
    frac_detmut = detected_mut / (mutfix * n * N)
    
    return af_ts, prob, frac_hetero, frac_detmut, homo_time, het_det_time, het_nondet_time, fixed_mut, het_det_list, het_age_dist, cp_fixed_time, cell_fixed_time


def sem_P(het_det_list):
    '''
    Calculating standard error of the mean considering individual mutations as independent samples from the distribution
    '''
    det_list, tot_list = [], []
    mut_num = len(het_det_list)
    for i in range(mut_num):
        det_list.append(het_det_list[i][0])
        tot_list.append(het_det_list[i][1] + het_det_list[i][2])
    mean_det, mean_tot = statistics.mean(det_list), statistics.mean(tot_list)
    ratio = mean_det / mean_tot
    sem_det, sem_tot = sem(det_list), sem(tot_list)
    sem_ratio = math.sqrt((sem_det/mean_det)**2 + (sem_tot/mean_tot)**2) * ratio
    return sem_ratio, sem_det, sem_tot
    
def sem_fixed(het_det_list):
    '''
    Calculating standard error of the mean considering individual mutations as independent samples from the distribution
    '''
    fixed_list = []
    mut_num = len(het_det_list)
    for i in range(mut_num):
        fixed_list.append(int(het_det_list[i][3] == 'fixed'))
    return sem(fixed_list) * mut_num

