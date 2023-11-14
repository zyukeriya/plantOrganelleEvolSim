import yaml
import multiprocessing
import math
import random
import numpy as np
import warnings
import time
import statistics
import copy
import sys
from scipy.stats import sem
import data_analysis
import PopState

########################################################################
#Overview: this script runs simulation of neutral mutation emergence and evolution in a plant genome. 
#The meristem structure is set for Zostera marina
#Input: a .yml config file
#Output: a table containing input parameters, the results of the simulation, and the error (one standard deviation) when applicable.
########################################################################


def calc_cells_left(N_curr, cell_names):
    '''
    The set of cells at the time of a mutation emergence N!=N_curr
    The cell names should be ordered accordingly
    '''
    cell_curr = cell_names[:N_curr]
    random.shuffle(cell_curr)
    return cell_curr


def calc_current_N(N, t_curr, branch_bneck=None, sex_bneck=None):
    '''
    Calculates the N_curr parameter at the time of a mutation emergence
    based on the current time and the bottlenecks.
    The bottlenecks are considered regular with a constant size of 1 and N // 2
    '''
    N_curr = N
    #if bottlenecks are applied
    if sex_bneck != None:
        #how many generations ago the bottleneck happened
        t_sex_ago = t_curr % sex_bneck
        #each generation +1 cell (prolifiration)
        N_curr = min(N, 1 + t_sex_ago)
    if branch_bneck != None:
        t_branch_ago = t_curr % branch_bneck
        if sex_bneck != None:
            if t_sex_ago - N + 1 >=t_branch_ago:
                N_curr = min(N, N // 2 + t_branch_ago)
        else:
            N_curr = min(N, N // 2 + t_branch_ago)
    return N_curr


def fixation_dyn(state, current_time=None, branch_bneck=None, sex_bneck=None):
    '''
    For the given state randomly identify the next state.
    When the state is a fixation or an extiction, stop (state.het==False)
    Returns t_ts, pop_ts - time and population state arrays of the same length 
    representing the segregation process of a single mutation in the experiment.
    '''
    
    curr_state = state
    curr_time = current_time
    pop_ts = [curr_state]
    t_ts = [1]
    #while not extict or fixed
    while curr_state.het:
        
        bneck, bneck_type = False, None
        prolif_type = 'by scheme'
        #if bottlenecks are applied
        if sex_bneck != None:
            if curr_time % sex_bneck == 0:
                bneck, bneck_type, prolif_type = True, 'sex', 'random'
            if curr_time % sex_bneck < curr_state.N:
                prolif_type = 'random'
        if branch_bneck != None and bneck_type == None:
            if curr_time % branch_bneck == 0:
                bneck, bneck_type = True, 'branchHalf'
        if curr_time != None:
            curr_time += 1
        
        #calculating next state
        next_state = curr_state.next_state(bneck=bneck, bneck_type=bneck_type, prolif_type=prolif_type)
        
        if next_state == pop_ts[-1]:
            t_ts[-1] += 1
        else:
            pop_ts.append(next_state)
            t_ts.append(1)
        
        curr_state = next_state
    
    return t_ts, pop_ts

def detection(u=5e-11, nrep=40, ncp=1, N=8, G=40, mutmax=100, branch_bneck=None, sex_bneck=None, subst_p=1, cell_subst=None, cell_names=None, contribution=None, error_p=None, dl=0.05, proc_name=None):
    '''
    Takes time of the emergence of the next mutation from binomial distribution and run segregation. 
    The result t_ts (time time series) and pop_ts (population time series) are combined for all mutations in the experiment and the homoplasmic time.
    If branch and sex bottlenecks are not None, the specific time points are on corresponding fixed distance (or taken from normal distribution with corresponding E).
    Bottlenecks in homogeneous moments are ignored.
    The analysis of the simulation results is performed inside this function (for the sake of computing time), 
    the result is put to the global dictionaries that collect information from all parallele processes.,
    '''
    
    #for random
    np.random.seed(random.randint(0, 100000))
    
    #We should start from a random period until a new mutation happens
    #N_start = N
    #define initial condition t=0 and insert into a time series (ts)
    t_ts=[0]
    state_zero = PopState.PopStatePlant(set(zip([tuple([0]*ncp)]*N, cell_names)), nrep, ncp, N, subst_p, cell_subst, error_p)
    pop_ts=[state_zero]
        
    counter_mutations=0
    
    # The probability that a mutation occurs in a wild-type population between 
    # two time steps equals one minus the probability that no mutation occurs. 
    # Here we assume that, for all mutational events, there is only 1 mutated 
    # replicon copy.
    #p=1-(1-u)**(n) #if all cell divisions are symmetric
    #p = (1 - (1 - u) ** n) * (subst_p + (1 - subst_p) / 2) #if we don't count the impact of bottlenecks
    #!here N is an even number! and branching doesn't happen while restoration from sexual reproduction is still on.
    #!!change here when changing the branching bottleneck size!!
    
    #fraction of generations with asymmetric division
    n = nrep * ncp
    frac_asymmetric = 1 - subst_p
    if sex_bneck != None:
        if branch_bneck != None:
            frac_asymmetric = (sex_bneck - N + 1) / sex_bneck * (branch_bneck - N//2) / branch_bneck * (1 - subst_p)
        else:
            frac_asymmetric = (sex_bneck - N + 1) / sex_bneck * (1 - subst_p)
    elif branch_bneck != None:
        frac_asymmetric = (branch_bneck - N//2) / branch_bneck * (1 - subst_p)
    p = (1 - (1 - u) ** n) * (1 - frac_asymmetric + frac_asymmetric / 2)
    
    #here the simulated process begins
    t_fixext = 0
    t_total = 0
    fixed_total = 0
    while counter_mutations < mutmax:
        # compute the time point of the next mutation
        t_wt=np.random.geometric(p=p)
        if t_wt < t_fixext:
            # search for time points when the next mutation given that fixation process
            # has finished
            warnings.warn(
                'Warning: a probability of incorrect assumption '+
                '(new mutation during ongoing fixation process) at time '+ str(t_wt))
            
            while t_wt <= t_fixext:
                t_wt += np.random.geometric(p=p)
        
        counter_mutations += 1
        t_total += t_wt
        
        #printing the progress
        new_frac = counter_mutations / mutmax
        prev_frac = (counter_mutations - 1) / mutmax
        if int(new_frac * 10) > int(prev_frac * 10):
            print('{0}: {1}% done'.format(proc_name, int(new_frac * 10) * 10))
        
        #add to t_ts and pop_ts
        if list(pop_ts[-1].state)[0][0][0] > 0:
            #fixation happend on the previous step
            pop_ts.append(state_zero)
            t_ts.append(t_wt - t_fixext - 1)
            fixed_total += 1
        else:
            #extinction on the previous step
            t_ts[-1] += (t_wt - t_fixext - 1)
        
        # fixation process
        #if no bottlenecks the new state is always [N-1, 1, 0,...,0]
        #else: [N_curr-1,1,0,...,0]
        
        N_curr = calc_current_N(N, t_total, branch_bneck, sex_bneck)
        #already shuffled, so we can assume the first cell has the mutation
        cells_left = calc_cells_left(N_curr, cell_names)
        mut_state = PopState.PopStatePlant(set(zip([tuple([1] + [0]*(ncp-1))] + [tuple([0]*ncp)]*(N_curr-1), cells_left)), nrep, ncp, N, subst_p, cell_subst, error_p)
        
        #compute the fate of a single mutation
        t_ts_fix, pop_ts_fix = fixation_dyn(state=mut_state, current_time=t_total, branch_bneck=branch_bneck, sex_bneck=sex_bneck)
       
        t_ts.extend(t_ts_fix)
        pop_ts.extend(pop_ts_fix)

        t_fixext=sum(t_ts_fix) - 1
    
    #sustract the last t_fixext time from total (for a fair calculation)
    t_ts[0] -= t_fixext
    
    print('{0} mutations got fixed.'.format(fixed_total))
    
    #simulation output analysis
    
    contribution_dic = dict(zip(cell_names, [1]*N))
    if type(contribution) != list:
        for cell in cell_names:
            if cell[:2] == 'L1':
                contribution_dic[cell] = contribution
       
    mutfix = mutmax // (nrep * ncp * N)
    
    global output_times_dict
    #for multiple dl
    if type(dl) == list:
        out_dic = {}
        for lim in dl:
            af_ts, prob, frac_hetero, frac_detmut, homo_time, het_det_time, het_nondet_time, fixed_mut, het_det_list, het_age_dist, cp_fixed_time, cell_fixed_time = data_analysis.det_nondet_het(t_ts, pop_ts, nrep, ncp, N, mutfix, dl=lim, contribution_dic=contribution_dic)
            out_dic[lim] = (homo_time, het_det_time, het_nondet_time, fixed_mut)
        output_times_dict[proc_name] = out_dic
    #for multiple contributions
    elif type(contribution) == list:
        out_dic = {}
        for contr in contribution:
            for cell in cell_names:
                if cell[:2] == 'L1':
                    contribution_dic[cell] = contr
            af_ts, prob, frac_hetero, frac_detmut, homo_time, het_det_time, het_nondet_time, fixed_mut, het_det_list, het_age_dist, cp_fixed_time, cell_fixed_time = data_analysis.det_nondet_het(t_ts, pop_ts, nrep, ncp, N, mutfix, dl=dl, contribution_dic=contribution_dic)
            out_dic[contr] = (homo_time, het_det_time, het_nondet_time, fixed_mut)
        output_times_dict[proc_name] = out_dic
    #for single dl and contribution
    else:
        af_ts, prob, frac_hetero, frac_detmut, homo_time, het_det_time, het_nondet_time, fixed_mut, het_det_list, het_age_dist, cp_fixed_time, cell_fixed_time = data_analysis.det_nondet_het(t_ts, pop_ts, nrep, ncp, N, mutfix, dl=dl, contribution_dic=contribution_dic)
        output_times_dict[proc_name] = (homo_time, het_det_time, het_nondet_time, fixed_mut)
    
    
    
    global output_het_age_dict
    het_age_year_dist = {}
    for age in het_age_dist:
        age_year = age // (G * N)
        het_age_year_dist.setdefault(age_year, 0)
        het_age_year_dist[age_year] += het_age_dist[age]
    output_het_age_dict[proc_name] = het_age_year_dist
    
    global output_het_det_list
    output_het_det_list[proc_name] = het_det_list
    
    global output_fixed_cp_age_list
    output_fixed_cp_age_list[proc_name] = cp_fixed_time
    
    global output_fixed_cell_age_list
    output_fixed_cell_age_list[proc_name] = cell_fixed_time


def detection_multiproc(u=5e-11, nrep=40, ncp=1, N=8, G=40, mutmax=100, proc_num=8, branch_bneck=None, sex_bneck=None, subst_p=1, cell_subst=None, cell_names=None, contribution=None, error_p=None, dl=0.05):
    '''
    Subdivides the simulation into processes for parralel calculations on multiple threads.
    Each process takes a share of the total number of novel mutations in the simulation (mutmax*nrep*ncp*N).
    Returns the final result of the simulation.
    '''
    proc_index_args = {}
    z = mutmax // proc_num
    res = mutmax % proc_num
    for i in range(proc_num):
        if i < res:
            mutmax_proc = z + 1
        else:
            mutmax_proc = z
        proc_index_args[i] = [u, nrep, ncp, N, G, mutmax_proc, branch_bneck, sex_bneck, subst_p, cell_subst, cell_names, contribution, error_p, dl]
    
    if __name__ == "__main__":  # confirms that the code is under main function
        manager = multiprocessing.Manager()
        global output_t_pop_ts_dict
        global output_times_dict
        global output_het_age_dict
        global output_het_det_list
        global output_fixed_cp_age_list
        global output_fixed_cell_age_list
        output_t_pop_ts_dict = manager.dict()
        output_times_dict = manager.dict()
        output_het_age_dict = manager.dict()
        output_het_det_list = manager.dict()
        output_fixed_cp_age_list = manager.dict()
        output_fixed_cell_age_list = manager.dict()
        
        procs = []
        
        for i in range(proc_num):
            proc_index_args[i].append('Process{0}'.format(i))
            proc = multiprocessing.Process(target=detection, args=tuple(proc_index_args[i]))
            procs.append(proc)
            proc.start()
    
        # complete the processes
        for proc in procs:
            proc.join()
        
        #Combing the results from individual processes
        
        #for multiple dl
        if type(dl) == list:
            times_dic = {}
            for i in range(proc_num):
                for dl in output_times_dict[proc_index_args[i][-1]]:
                    times_dic.setdefault(dl, {'homo': 0, 'hetero det': 0, 'hetero non-det': 0, 'fixed mut': 0})
                    homo_time, het_det_time, het_nondet_time, fixed_mut = output_times_dict[proc_index_args[i][-1]][dl]
                    times_dic[dl]['homo'] += homo_time
                    times_dic[dl]['hetero det'] += het_det_time
                    times_dic[dl]['hetero non-det'] += het_nondet_time
                    times_dic[dl]['fixed mut'] += fixed_mut
        #for multiple contributions
        elif type(contribution) == list:
            times_dic = {}
            for i in range(proc_num):
                for contr in output_times_dict[proc_index_args[i][-1]]:
                    times_dic.setdefault(contr, {'homo': 0, 'hetero det': 0, 'hetero non-det': 0, 'fixed mut': 0})
                    homo_time, het_det_time, het_nondet_time, fixed_mut = output_times_dict[proc_index_args[i][-1]][contr]
                    times_dic[contr]['homo'] += homo_time
                    times_dic[contr]['hetero det'] += het_det_time
                    times_dic[contr]['hetero non-det'] += het_nondet_time
                    times_dic[contr]['fixed mut'] += fixed_mut
        
        #for single dl and contribution
        else:
            times_dic = {'homo': 0, 'hetero det': 0, 'hetero non-det': 0, 'fixed mut': 0}
            for i in range(proc_num):
                homo_time, het_det_time, het_nondet_time, fixed_mut = output_times_dict[proc_index_args[i][-1]]
                times_dic['homo'] += homo_time
                times_dic['hetero det'] += het_det_time
                times_dic['hetero non-det'] += het_nondet_time
                times_dic['fixed mut'] += fixed_mut
        
        #for heteroplasmy age distribution
        het_age_dist_joined = {}
        for i in range(proc_num):
            het_age_dist = output_het_age_dict[proc_index_args[i][-1]]
            for age in het_age_dist:
                het_age_dist_joined.setdefault(age, 0)
                het_age_dist_joined[age] += het_age_dist[age]
        
        #for error calculation
        het_det_list_joined = []
        fixed_cp_age_list_joined = []
        fixed_cell_age_list_joined = []
        for i in range(proc_num):
            het_det_list_joined.extend(output_het_det_list[proc_index_args[i][-1]])
            fixed_cp_age_list_joined.extend(output_fixed_cp_age_list[proc_index_args[i][-1]])
            fixed_cell_age_list_joined.extend(output_fixed_cell_age_list[proc_index_args[i][-1]])
        
        return times_dic, het_age_dist_joined, het_det_list_joined, fixed_cp_age_list_joined, fixed_cell_age_list_joined


 
######################### Main #########################################

config_file = sys.argv[1]

with open(config_file, 'r') as ymlfile:
    cfgs = yaml.safe_load(ymlfile)

print(cfgs)

param_names = list(cfgs['input'])

#each element of the list is a dictionary of params for the detection_multiproc function
#default dict
default_param_dic = {}
default_param_dic['nrep'] = cfgs['input']['nrep']
default_param_dic['ncp'] = cfgs['input']['ncp']
default_param_dic['N'] = cfgs['input']['N']
default_param_dic['mutmax'] = cfgs['input']['mutfix'] #before running needs to be multiplied by N, nrep, and ncp
default_param_dic['subst_p'] = cfgs['input']['subst_p']
default_param_dic['G'] = cfgs['input']['G']
default_param_dic['error_p'] = cfgs['input']['error_p']
default_param_dic['u'] = cfgs['input']['u']
default_param_dic['dl'] = cfgs['input']['dl']
default_param_dic['cell_names'] = cfgs['input']['cell_names']
default_param_dic['contribution'] = cfgs['input']['contribution']
default_param_dic['cell_subst'] = cfgs['input']['cell_subst']
default_param_dic['stratification_prop'] = cfgs['input']['stratification_prop']
default_param_dic['proc_num'] = cfgs['run']['proc_num']

detection_multiproc_params = []
#here the grids are not combining, but applied individually to the default parameters
#bottleneck regimes are later at a separate loop
for param in param_names:
    if param in cfgs['input_grids']:
        for param_val in cfgs['input_grids'][param]:
            new_param_dic = copy.deepcopy(default_param_dic)
            new_param_dic[param] = param_val
            #if ncp is on grid, then ncp*nrep = constant
            if param == 'ncp':
                new_param_dic['nrep'] = new_param_dic['nrep'] // new_param_dic['ncp']
            detection_multiproc_params.append(new_param_dic)
if len(detection_multiproc_params) == 0:
    detection_multiproc_params.append(default_param_dic)

#print output
edit_table = cfgs['output']['edit_table']
out_file = open(cfgs['output']['out_table'], edit_table)
if edit_table == 'w':
    out_file.write('Number of replicons per organelle\tNumber of organelles per cell\tNumber of cells\tGenerations per year\tExpected fixed mutations\tSpontaneous mutation rate (u)\t' +
    'Symmetric division probability\tBottleneck regime\tPartitioning error\tStratification proportion\tContribution proportion\tDetection limit\t' + 
    'Heteroplasmy probability\tExpected heteroplasmy sites\tNumber of threads\tCalculation time (min)\tTotal time\tTotal Detectable time\tMutation fixed in 243300 years per bp\t' +
    'Standard error heteroplasmy probability\tStandard error Expected heteroplasmy\tStandard error Ttotal\tStandard error Tdet\tStandard error fixed in 243300 years per bp\t' +
    'Heteroplasmy age (years)\tCount\tTime to first organellar fixation\tStandard error first organellar fixation\tTime to first cellular fixation\tStandard error first cellular fixation\tComment\n')

#parameter grids
for param_set in detection_multiproc_params:
    strat_prop = param_set['stratification_prop']
    #stratification_prop is not an input parameter of the simulation itself
    del param_set['stratification_prop']
    
    for cell in param_set['cell_subst']:
        if cell[:2] == 'L1':
            for i in range(len(param_set['cell_subst'][cell][0])):
                if param_set['cell_subst'][cell][0][i][:2] == 'L1':
                    param_set['cell_subst'][cell][1][i] = strat_prop
    gen_year = param_set['G'] * param_set['N']
    param_set['mutmax'] = param_set['mutmax'] * param_set['N'] * param_set['nrep'] * param_set['ncp'] #calculate it here because the parameters could change in grids
    for [branch_bneck, sex_bneck, bneck_regime, recalc_u, recalc_error_p] in cfgs['input']['bneck_regime']:
        print('Bottekneck regime: ', branch_bneck, sex_bneck, bneck_regime, recalc_u)
        if branch_bneck != None:
            branch_bneck = gen_year // branch_bneck
        if sex_bneck != None:
            sex_bneck = gen_year * sex_bneck
        if recalc_u != None:
            param_set['u'] = recalc_u
        if recalc_error_p != None:
            param_set['error_p'] = recalc_error_p
        param_set['branch_bneck'] = branch_bneck
        param_set['sex_bneck'] = sex_bneck
        
        print(param_set)
        
        #simulation run
        start = time.time()
        times_dic, het_age_dist, het_det_list_joined, fixed_cp_age_list_joined, fixed_cell_age_list_joined = detection_multiproc(**param_set)
        end = time.time()
        
        #stat #can be used only for single dl and contributions
        sem_P, sem_Tdet, sem_Ttotal = data_analysis.sem_P(het_det_list_joined) 
        sem_fixed_total = data_analysis.sem_fixed(het_det_list_joined)
        
        #time until fixation (should be used only for single dl or sequencing scheme grid)
        Tcp, Tcell, sem_Tcp, sem_Tcell = statistics.mean(fixed_cp_age_list_joined), statistics.mean(fixed_cell_age_list_joined), sem(fixed_cp_age_list_joined), sem(fixed_cell_age_list_joined)
        
        #print output
        mutfix = cfgs['input']['mutfix']
        divergence_time = cfgs['input']['divergence_time']
        total_pos = cfgs['input']['total_pos']
        #strat_prop = cfgs['input']['stratification_prop']
        comment = cfgs['output']['comment']
        
        #for multiple dl
        if type(param_set['dl']) == list:
            for dl_val in param_set['dl']:
                homo_time, het_det_time, het_nondet_time, fixed_mut = times_dic[dl_val]['homo'], times_dic[dl_val]['hetero det'], times_dic[dl_val]['hetero non-det'], times_dic[dl_val]['fixed mut']
                total_time = homo_time + het_det_time + het_nondet_time
                prob = het_det_time / total_time
                fixed_mut_accum = fixed_mut * gen_year * divergence_time / total_time #if for mt, then * 121502, here per base pair
                sem_fixed_mut_accum = sem_fixed_total * gen_year * divergence_time / total_time #if for mt, then * 121502, here per base pair
                
                out_file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\t{16}\t{17}\t{18}\t{19}\t{20}\t{21}\t{22}\t{23}\t{24}\t{25}\t{26}\t{27}\t{28}\t{29}\t{30}\n'.format(
                param_set['nrep'], param_set['ncp'], param_set['N'], param_set['G'], mutfix, param_set['u'],
                param_set['subst_p'], bneck_regime, param_set['error_p'], strat_prop, param_set['contribution'], dl_val,
                prob, prob*total_pos, param_set['proc_num'], (end - start) / 60, total_time, het_det_time, fixed_mut_accum,
                '-', '-', '-', '-', sem_fixed_mut_accum, '-', '-', '-', '-', '-', '-', comment))
                
        #for multiple contributions
        elif type(param_set['contribution']) == list:
            for contr in param_set['contribution']:
                homo_time, het_det_time, het_nondet_time, fixed_mut = times_dic[contr]['homo'], times_dic[contr]['hetero det'], times_dic[contr]['hetero non-det'], times_dic[contr]['fixed mut']
                total_time = homo_time + het_det_time + het_nondet_time
                prob = het_det_time / total_time
                fixed_mut_accum = fixed_mut * gen_year * divergence_time / total_time #if for mt, then * 121502, here per base pair
                sem_fixed_mut_accum = sem_fixed_total * gen_year * divergence_time / total_time #if for mt, then * 121502, here per base pair
                
                out_file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\t{16}\t{17}\t{18}\t{19}\t{20}\t{21}\t{22}\t{23}\t{24}\t{25}\t{26}\t{27}\t{28}\t{29}\t{30}\n'.format(
                param_set['nrep'], param_set['ncp'], param_set['N'], param_set['G'], mutfix, param_set['u'],
                param_set['subst_p'], bneck_regime, param_set['error_p'], strat_prop, contr, param_set['dl'],
                prob, prob*total_pos, param_set['proc_num'], (end - start) / 60, total_time, het_det_time, fixed_mut_accum,
                '-', '-', '-', '-', sem_fixed_mut_accum, '-', '-', '-', '-', '-', '-', comment))
                
        #for single dl and contribution
        else:
            homo_time, het_det_time, het_nondet_time, fixed_mut = times_dic['homo'], times_dic['hetero det'], times_dic['hetero non-det'], times_dic['fixed mut']
            total_time = homo_time + het_det_time + het_nondet_time
            prob = het_det_time / total_time
            fixed_mut_accum = fixed_mut * gen_year * divergence_time / total_time #if for mt, then * 121502, here per base pair
            sem_fixed_mut_accum = sem_fixed_total * gen_year * divergence_time / total_time #if for mt, then * 121502, here per base pair
            
            if cfgs['output']['het_age']:
                for age in het_age_dist:
                    out_file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\t{16}\t{17}\t{18}\t{19}\t{20}\t{21}\t{22}\t{23}\t{24}\t{25}\t{26}\t{27}\t{28}\t{29}\t{30}\n'.format(
                    param_set['nrep'], param_set['ncp'], param_set['N'], param_set['G'], mutfix, param_set['u'],
                    param_set['subst_p'], bneck_regime, param_set['error_p'], strat_prop, param_set['contribution'], param_set['dl'],
                    prob, prob*total_pos, param_set['proc_num'], (end - start) / 60, total_time, het_det_time, fixed_mut_accum,
                    sem_P, sem_P*total_pos, sem_Ttotal, sem_Tdet, sem_fixed_mut_accum, age, het_age_dist[age], Tcp, sem_Tcp, Tcell, sem_Tcell, comment))
            else:
                out_file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\t{16}\t{17}\t{18}\t{19}\t{20}\t{21}\t{22}\t{23}\t{24}\t{25}\t{26}\t{27}\t{28}\t{29}\t{30}\n'.format(
                param_set['nrep'], param_set['ncp'], param_set['N'], param_set['G'], mutfix, param_set['u'],
                param_set['subst_p'], bneck_regime, param_set['error_p'], strat_prop, param_set['contribution'], param_set['dl'],
                prob, prob*total_pos, param_set['proc_num'], (end - start) / 60, total_time, het_det_time, fixed_mut_accum,
                sem_P, sem_P*total_pos, sem_Ttotal, sem_Tdet, sem_fixed_mut_accum, '-', '-', Tcp, sem_Tcp, Tcell, sem_Tcell, comment))

out_file.close()        
