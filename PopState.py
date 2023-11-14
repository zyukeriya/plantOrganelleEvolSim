import random
import itertools
from bisect import bisect_left

########################################################################
#File with the class PopState description
#The Population consists of DNA replicons inside plastids inside stem cells. 
#For mitocondria n_cp == 1 can be used, thereby reducing by one hierarchical level
########################################################################


class PopStatePlant:
    '''
    Takes a set of parameters to initiate. 'State' is a set of (N_curr) cells with elements representing genotypes of individual cell and the cell names.
    Inside each cell a genotype is represented as a tuple of individual chloroplast genotypes
    '''
    def __init__(self, state, nrep, ncp, N, subst_p, cell_subst, error_p):
        self.state = state
        self.sort_state()
        self.N_curr = len(state)
        self.nrep = nrep
        self.ncp = ncp
        self.N = N
        self.subst_p = subst_p
        #the cell names which I get as input is in fact stratifications. I put keys into cell_names and the whole dictionary to stratification
        self.cell_names = set(cell_subst.keys())
        self.stratification = cell_subst
        self.error_p = error_p
        self.is_heterozygote()
    
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.state == other.state
    
    def __hash__(self):
        return hash(tuple(sorted(list(self.state), key=lambda x: x[1])))
    
    def sort_state(self):
        state_list = list(self.state)
        sorted_state_list = []
        for cell in state_list:
            sorted_state_list.append((tuple(sorted(list(cell[0]))), cell[1]))
        self.state = set(sorted_state_list)
    
    def is_heterozygote(self):
        if set([sum(x[0]) for x in self.state]) == {0} or set([sum(x[0]) for x in self.state]) == {self.nrep * self.ncp}:
            self.het = False
        else:
            self.het = True
    
    
    def random_bottleneck(self, bneck_type='branchHalf'):
        '''
        Returns a modified state (set) after a random choice of cells according to the bottleneck type
        If N_curr != N then no bottleneck.
        L1, and L3 are two (as in momocots, although in that case the corpus is called L2) meristematic layers. For dicots ir is usually L1, L2, and L3
        branchHalf - either one or another half of the meristem
        For sex: one cell from not L1
        '''
        #no branching if the set of cells is not full
        if self.N_curr != self.N and bneck_type != 'sex':
            return self.state
        
        #divide into layers
        state_nonL1 = []
        state_vertical, state_horizontal = {}, {}
        for cell in list(self.state):
            if cell[1][:2] != 'L1':
                state_nonL1.append(cell)
            state_horizontal.setdefault(cell[1][:2], []).append(cell)
            state_vertical.setdefault(cell[1][2:], []).append(cell)
        if bneck_type == 'sex':
            return set(random.sample(state_nonL1, 1))
        elif bneck_type == 'branchHalf':
            #only one or another half
            #which half
            half = random.randint(0,1)
            #works only for N divisible by 4, scheme: a block of two rows in each layer
            in_one_row = self.N // 4
            new_state = []
            for cell in list(self.state):
                if half * in_one_row + in_one_row >= int(cell[1][2:]) > half * in_one_row:
                    new_state.append(cell)
            return set(new_state)
    
    def division_dice(self, divtype='symmetric', bneck=False, bneck_type=None, prolif_type='random'):
        '''
        random choice of two cells (one to divide, one to die[optional]) -> random segregation -> return new state (set, not a PopStatePlant object)
        if divtype == 'proliferation' the second cell is not removed from the population. 
        Proliferation by_scheme applies a certain order for L1 and L3 cells reappearence in the set (stratification between the layers is conserved), othervise the proliferation is random.
        if divtype == 'asymmetric' the second cell is not removed from the population; only one daughter cell stays in the population
        bneck_size is the number of remaining cells
        '''
        if bneck and self.N_curr == self.N:
            state_list = list(self.random_bottleneck(bneck_type=bneck_type))
            divtype = 'proliferation'
        else:
            state_list = list(self.state)
        
        if divtype == 'proliferation' and prolif_type == 'by scheme':
            return self.division_poliferation_scheme(state_list)
        
        i = random.randint(0, len(state_list) - 1)
        #genotype
        c1_mut, c1_name = state_list[i] #to divide
        #new genotypes
        replicated = []
        for mut_cp in list(c1_mut):
            new_cp1, new_cp2 = segregation_result_genotypes(mut_cp, self.nrep)
            #new_cp1, new_cp2 = segregation_relaxed_result_genotypes(mut_cp, self.nrep)
            replicated.extend([new_cp1, new_cp2])
        #random chloroplast segregation
        if self.error_p == None:
            random.shuffle(replicated)
            new_c1, new_c2 = tuple(replicated[:self.ncp]), tuple(replicated[self.ncp:])
        #active chloroplast segregation
        else:
            new_c1, new_c2 = segregation_active(replicated, error_p=self.error_p)
        #create new state set
        state_list[i] = (new_c1, c1_name)
        if divtype == 'asymmetric':
            return set(state_list)
        if divtype == 'proliferation':
            names = set([x[1] for x in state_list])
            missed_names = self.cell_names.difference(names)
            c2_name = random.choice(list(missed_names))
            state_list.append((new_c2, c2_name))
            return set(state_list)
        if divtype == 'symmetric':
            #with stratification
            state_dic = {x[1]: x for x in state_list}
            #in case we have weights as well
            if isinstance(self.stratification[c1_name][0], list):
                c2_mut, c2_name = state_dic[random.choices(self.stratification[c1_name][0], weights=self.stratification[c1_name][1], k=1)[0]] #to delete
            else:
                c2_mut, c2_name = state_dic[random.sample(self.stratification[c1_name], 1)[0]] #to delete
            state_dic[c2_name] = (new_c2, c2_name)
            return set(state_dic.values())
    
    def division_poliferation_scheme(self, state_list):
        '''
        Works for any L1 = L3 or L1 = L3+1 in the number of cells
        '''
        cells_L1 = list(filter(lambda x: x[1][:2] == 'L1', state_list))
        cells_L3 = list(filter(lambda x: x[1][:2] == 'L3', state_list))
        all_names_L1 = set([x for x in self.cell_names if x[:2] == 'L1'])
        all_names_L3 = set([x for x in self.cell_names if x[:2] == 'L3'])
        if len(cells_L1) <= len(cells_L3):
            i = random.randint(0, len(cells_L1) - 1)
            c1_mut, c1_name = cells_L1[i] #to divide
            names = set([x[1] for x in cells_L1])
            missed_names = all_names_L1.difference(names)
            c2_name = random.choice(list(missed_names))
            del cells_L1[i]
        else:
            i = random.randint(0, len(cells_L3) - 1)
            c1_mut, c1_name = cells_L3[i] #to divide
            names = set([x[1] for x in cells_L3])
            missed_names = all_names_L3.difference(names)
            c2_name = random.choice(list(missed_names))
            del cells_L3[i]
        state_list = cells_L1 + cells_L3
        #new genotypes
        replicated = []
        for mut_cp in list(c1_mut):
            new_cp1, new_cp2 = segregation_result_genotypes(mut_cp, self.nrep)
            #new_cp1, new_cp2 = segregation_relaxed_result_genotypes(mut_cp, self.nrep)
            replicated.extend([new_cp1, new_cp2])
        #random chloroplast segregation
        if self.error_p == None:
            random.shuffle(replicated)
            new_c1, new_c2 = tuple(replicated[:self.ncp]), tuple(replicated[self.ncp:])
        #active chloroplast segregation
        else:
            new_c1, new_c2 = segregation_active(replicated, error_p=self.error_p)
        #change the state list
        state_list.append((new_c1, c1_name))
        state_list.append((new_c2, c2_name))
        return set(state_list)
                
    
    def next_state(self, bneck=False, bneck_type=None, prolif_type='random'):
        '''
        Calculate next state, return a PopStatePlant object
        Asymmetric or symmetric is random according to subst_p value
        prolif_type can be random or by scheme
        '''
        if self.N_curr < self.N // 2:
            prolif_type='random'
        if self.N_curr != self.N:
            new_state_set = self.division_dice(divtype='proliferation', bneck=bneck, bneck_type=bneck_type, prolif_type=prolif_type)
            return PopStatePlant(new_state_set, self.nrep, self.ncp, self.N, self.subst_p, self.stratification, self.error_p)
        divtype = random.choices(['symmetric', 'asymmetric'], weights=[self.subst_p, 1 - self.subst_p], k=1)[0]
        new_state_set = self.division_dice(divtype=divtype, bneck=bneck, bneck_type=bneck_type, prolif_type=prolif_type)
        return PopStatePlant(new_state_set, self.nrep, self.ncp, self.N, self.subst_p, self.stratification, self.error_p)


def segregation_result_genotypes(mut, n):
    '''
    Takes the dividing cell nuber of mutant alleles and the total number of replicons.
    Returns two numbers of mutant alleles of the two daughter cells
    '''
    alleles = [1] * (2 * mut)
    alleles.extend([0] * (2 * (n - mut)))
    mut1 = sum(random.sample(alleles, k=n))
    mut2 = 2 * mut - mut1
    return mut1, mut2


def segregation_active(cp_pairs, error_p=0.05):
    '''
    before - segregation_active2
    Second attempt for active partitioning. Normally sister chloroplast go to different daughter groups, 
    but with p=error_p two pars stay together.
    This method is exact but works for even number of cp per cell only.
    The input list of cps is has the pairs together one after another
    '''
    #step1: group pairs in quads
    paired_list = [[cp_pairs[i], cp_pairs[i+1]] for i in range(0, len(cp_pairs), 2)]
    random.shuffle(paired_list)
    quad_list = [[paired_list[i], paired_list[i+1]] for i in range(0, len(paired_list), 2)]
    #step2: for each quad deside whether they segregate correctly or whith an error
    group1, group2 = [], []
    for quad in quad_list:
        seg_type = random.choices(['error', 'exact'], weights=[error_p, 1 - error_p], k=1)[0]
        if seg_type == 'exact':
            group1.extend([quad[0][0], quad[1][0]])
            group2.extend([quad[0][1], quad[1][1]])
        elif seg_type == 'error':
            group1.extend(quad[0])
            group2.extend(quad[1])
    return(tuple(group1), tuple(group2))
