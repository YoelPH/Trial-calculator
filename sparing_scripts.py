import numpy as np
import lymph

def sample_from_flattened(flattened_samples, num_samples=100, spaced=False, step_size = None):
    """
    Function to sample from a flattened set of MCMC samples.

    Parameters:
    - flattened_samples (np.ndarray): The flattened MCMC samples (2D array, shape [num_samples, num_parameters]).
    - num_samples (int): The number of samples to select (default is 100).
    - spaced (bool): If True, select evenly spaced samples instead of random ones (default is False).

    Returns:
    - np.ndarray: The selected samples.
    """
    if spaced:
        # Select evenly spaced samples
        if step_size is None:
            step_size = flattened_samples.shape[0] // num_samples
        return flattened_samples[::step_size][:num_samples]
    else:
        # Select random samples without replacement
        indices = np.random.choice(flattened_samples.shape[0], size=num_samples, replace=False)
        return flattened_samples[indices]


def change_base(
    number: int,
    base: int,
    reverse: bool = False,
    length = None
) -> str:
    """Convert an integer into another base.

    Args:
        number: Number to convert
        base: Base of the resulting converted number
        reverse: If true, the converted number will be printed in reverse order.
        length: Length of the returned string. If longer than would be
            necessary, the output will be padded.

    Returns:
        The (padded) string of the converted number.
    """
    if number < 0:
        raise ValueError("Cannot convert negative numbers")
    if base > 16:
        raise ValueError("Base must be 16 or smaller!")
    elif base < 2:
        raise ValueError("There is no unary number system, base must be > 2")

    convertString = "0123456789ABCDEF"
    result = ''

    if number == 0:
        result += '0'
    else:
        while number >= base:
            result += convertString[number % base]
            number = number//base
        if number > 0:
            result += convertString[number]

    if length is None:
        length = len(result)
    elif length < len(result):
        length = len(result)

    pad = '0' * (length - len(result))

    if reverse:
        return result + pad
    else:
        return pad + result[::-1]
    

def risk_sampled(samples, model, t_stage, midline_extension=None, given_diagnoses=None, central=False):
    """
    Samples risk estimates from a model using provided parameter samples.

    Iterates over a set of parameter samples, sets the model parameters for each sample,
    and computes the risk for the given T stage and optional diagnosis information.
    Returns both the array of sampled risks and their mean.
    NOTE: The samples must have the correct parameter structure expected by the model.
    In the future we will handle this differently.

    Args:
        samples (array-like): An iterable of parameter samples, where each sample is a sequence of parameter values.
        model (object): A model object with `set_params` and `risk` methods.
        t_stage (any): The T stage input to be passed to the model's risk calculation.
        midline_extension (optional): Additional parameter for risk calculation, default is None.
        given_diagnoses (optional): Diagnosis information for risk calculation, default is None.
        central (bool, optional): Whether to use central calculation, default is False.

    Returns:
        tuple: 
            - sampled_risks (numpy.ndarray): Array of risk values computed for each sample.
            - mean_risk (numpy.ndarray or float): Mean of the sampled risks across all samples.
    """
    sampled_risks = np.zeros(shape=(len(samples),64,64), dtype=float)
    for i, sample in enumerate(samples):
        params = {'mixing': sample[0],
        'ipsi_primarytoI_spread': sample[1],
        'ipsi_primarytoII_spread': sample[2],
        'ipsi_primarytoIII_spread': sample[3],
        'ipsi_primarytoIV_spread': sample[4],
        'ipsi_primarytoV_spread': sample[5],
        'ipsi_primarytoVII_spread': sample[6],
        'contra_primarytoI_spread': sample[7],
        'contra_primarytoII_spread': sample[8],
        'contra_primarytoIII_spread': sample[9],
        'contra_primarytoIV_spread': sample[10],   
        'contra_primarytoV_spread': sample[11],
        'contra_primarytoVII_spread': sample[12],
        'ItoII_spread': sample[13],
        'IItoIII_spread': sample[14],
        'IIItoIV_spread': sample[15],
        'IVtoV_spread': sample[16],
        'late_p': sample[17]}
        model.set_params(**params)
        sampled_risks[i] = model.risk(t_stage = t_stage, given_diagnoses = given_diagnoses, midline_extension = midline_extension, central = central) 
    mean_risk = sampled_risks.mean(axis = 0)
    return sampled_risks, mean_risk


def ci_single(sampled_risks, level=0.95):
    """
    Calculate the credibility interval for a given set of sampled risks.

    Args:
        sampled_risks (np.ndarray): Array of sampled risks.
        level (float): credibility level (default is 0.95).

    Returns:
        np.ndarray: Lower and upper bounds of the credibility interval.
    """
    lower = (1 - level) / 2 * 100
    upper = 100 - lower
    ci = np.percentile(sampled_risks, [lower, upper])
    return ci


def ci_multiple(sampled_risks_set, level=0.95):
    """
    Calculate the credibility intervals for a set of sampled risks.

    Args:
        sampled_risks (np.ndarray): Array of sampled risks for multiple cases.
        level (float): Credibility level (default is 0.95).

    Returns:
        np.ndarray: Array of lower and upper bounds for each set of sampled risks.
    """
    lower = (1 - level) / 2 * 100
    upper = 100 - lower
    ci = np.zeros((len(sampled_risks_set), 2))
    for index in range(len(sampled_risks_set)):
        ci[index] = np.percentile(sampled_risks_set[index], [lower, upper])
    return ci


def get_risks_by_side(risks, state_list, lnls):
    ipsi = {lnl: risks[state_list[:, i] == 1].sum() for i, lnl in enumerate(lnls)}
    contra = {lnl: risks.T[state_list[:, i] == 1].sum() for i, lnl in enumerate(lnls)}
    return ipsi, contra

def get_lnl_indices(lnl_names, lnls):
    ipsi, contra = [], []
    for name in lnl_names:
        side, lnl = name.split()
        (ipsi if side == 'ipsi' else contra).append(lnls.index(lnl))
    return ipsi, contra

def get_state_indices(state_list, indices):
    combined = []
    for idx in indices:
        combined.extend(np.where(state_list[:, idx] == 1)[0])
    return np.unique(combined)


#NOTE This function adds the highest LNL when the threshold is surpassed by the upper CI bound. However the risk is not recalculated! Thus the treatment that is proposed is already with the next LNL included, the risk is therefor not the risk when this LNL is also treated, but the risk when we treat one LNL less.
def levels_to_spare_old(threshold, model, risks, sampled_risks):
    """
    Computes which LNLs to irradiate given the threshold, model and the risk of each state.
    NOTE: This function adds the highest LNL when the threshold is surpassed by the upper CI bound.
    However, the risk is not recalculated after this inclusion, meaning the returned risk may reflect
    one fewer LNL than actually treated.

    Args:
        threshold (float): Risk threshold we want to apply
        model (lymph.Unilateral): lymph.unilateral object with fully analyzed patients
        risks (ndarray): Array with the risk of each state
        sampled_risks (ndarray): Sampled risk array for uncertainty estimation

    Returns:
        spared_lnls (list), total_risk (float), ranked_combined (list), treated_lnls (list),
        treated_array (ndarray), treated_ipsi (list), treated_contra (list), sampled_total_risks (ndarray)
    """
    if threshold <= 0:
        raise ValueError("Threshold must be larger than zero")

    state_list = np.zeros((2**6,6))
    for i in range(2**6):
        state_list[i] = [
            int(digit) for digit in change_base(i, 2, length=6)
        ]
    lnls = ['I', 'II', 'III', 'IV', 'V', 'VII']
        
    # Get per-LNL risks
    ipsi_risks, contra_risks = get_risks_by_side(risks, state_list, lnls)
    combined_risks = {f'ipsi {k}': v for k, v in ipsi_risks.items()}
    combined_risks.update({f'contra {k}': v for k, v in contra_risks.items()})
    ranked_combined = sorted(combined_risks.items(), key=lambda x: x[1])
    contra_lnl_indices = []
    ipsi_lnl_indices = []
    treated_ipsi = []
    treated_contra = []
    total_risk_new = 0
    sampled_total_risks_new = np.zeros(sampled_risks.shape[0])
    treated_array = np.ones(len(ranked_combined))  # 12 LNLs (6 ipsi, 6 contra)
    looper = 1

    while total_risk_new < threshold:
        sampled_total_risks = sampled_total_risks_new
        total_risk = total_risk_new
        if ipsi_lnl_indices != []:
            treated_array[ipsi_lnl_indices] = 0
        if contra_lnl_indices != []:
            treated_array[np.array(contra_lnl_indices)+6] = 0
        lnls_of_interest = ranked_combined[0:looper]
        lnls_of_interest_names = [t[0] for t in lnls_of_interest]
        contra_lnl_indices = []
        ipsi_lnl_indices = []
        for i,lnl_looper in enumerate(lnls_of_interest_names):
            contra_lnl_indices.append(np.where(np.array(lnls) == lnls_of_interest_names[i].split()[1])[0][0]) if lnl_looper.split()[0] == 'contra' else ipsi_lnl_indices.append(np.where(np.array(lnls) == lnls_of_interest_names[i].split()[1])[0][0])
        indices_list_contra = []
        indices_list_ipsi = []
        for index in contra_lnl_indices:
            condition_contra = (state_list[:, index] == 1)
            indices_contra = np.where(condition_contra)[0]
            indices_list_contra.extend(indices_contra)
            unique_contra = np.unique(indices_list_contra)
        for index in ipsi_lnl_indices:
            condition_ipsi = (state_list[:, index] == 1)
            indices_ipsi = np.where(condition_ipsi)[0]
            indices_list_ipsi.extend(indices_ipsi)
            unique_ipsi = np.unique(indices_list_ipsi)    
        if len(ipsi_lnl_indices) == 0:
            total_risk_new = risks.T[unique_contra].sum()
            sampled_total_risks_new = sampled_risks.transpose((0,2,1))[:,unique_contra].sum(axis = (1,2))
        elif len(contra_lnl_indices) == 0:
            total_risk_new = risks[unique_ipsi].sum()
            sampled_total_risks_new = sampled_risks[:,unique_ipsi].sum(axis = (1,2))
        else:
            total_risk_new = 0
            sampled_total_risks_new = np.zeros(sampled_risks.shape[0])
            total_risk_new += risks[unique_ipsi].sum()
            total_risk_new += risks.T[unique_contra][:,[np.setdiff1d(np.array(range(64)),unique_ipsi)]].sum()
            sampled_total_risks_new += sampled_risks[:,unique_ipsi].sum(axis = (1,2))
            sampled_total_risks_new += sampled_risks.transpose((0,2,1))[:,unique_contra][:,:,list(np.setdiff1d(np.array(range(64)),unique_ipsi))].sum(axis = (1,2))
        if ci_single(sampled_total_risks)[1] > threshold:
            spared_lnls = lnls_of_interest[:-2]
            treated_lnls = ranked_combined[looper-2:]
        else:
            spared_lnls = lnls_of_interest[:-1]
            treated_lnls = ranked_combined[looper-1:]
        looper += 1
    for to_treat in treated_lnls:
        if to_treat[0].split()[0] == 'ipsi':
            treated_ipsi.append(to_treat[0].split()[1])
        else: 
            treated_contra.append(to_treat[0].split()[1])
    return spared_lnls, total_risk, ranked_combined, treated_lnls, treated_array, treated_ipsi, treated_contra,sampled_total_risks
    
def levels_to_spare(threshold, model, mean_risks, sampled_risks, ci=False):
    """
    Determine the levels of lymph nodes to spare based on a risk threshold.
    This function evaluates the risks associated with ipsilateral (ipsi) and 
    contralateral (contra) lymph node levels (LNLs) and determines which levels 
    can be spared while keeping the total risk below a specified threshold.
    Parameters:
    -----------
    threshold : float
        The maximum allowable total risk for sparing lymph node levels.
    model : object
        A model object containing the state list and lymph node level (LNL) information.
    risks_mean : numpy.ndarray
        Array of risks associated with each state in the model.
    sampled_risks : numpy.ndarray
        Array of sampled risks for uncertainty estimation, with shape 
        (num_samples, num_states, num_states).
    ci : bool, optional
        If True, confidence intervals are used to determine sparing decisions. 
        Default is False.
    Returns:
    --------
    spared_lnls : list of tuples
        List of spared lymph node levels and their associated risks, sorted by risk.
    total_risk_new : float
        The total risk after sparing the selected lymph node levels.
    ranked_combined : list of tuples
        List of all lymph node levels and their associated risks, sorted by risk.
    treated_lnls : list of tuples
        List of treated lymph node levels and their associated risks.
    treated_array : numpy.ndarray
        Array indicating which states are treated (1 for treated, 0 for spared).
    treated_ipsi : list of str
        Names of treated ipsilateral lymph node levels.
    treated_contra : list of str
        Names of treated contralateral lymph node levels.
    sampled_total_risks_new : numpy.ndarray
        Array of sampled total risks after sparing the selected lymph node levels.
    Notes:
    ------
    - The function iteratively evaluates lymph node levels to spare, starting 
      from the lowest risk levels, until the total risk exceeds the threshold.
    - If `ci` is True, confidence intervals are used to refine the sparing 
      decision.
    """
    if threshold <= 0:
        raise ValueError("Threshold must be larger than zero")
    if isinstance(model, lymph.models.Midline):
        lnls = list(model.noext.ipsi.graph.lnls.keys())
    else:
        raise TypeError("Model must be an instance of lymph.models.Midline")

    state_list = np.zeros((2**len(lnls), len(lnls)))
    for i in range(2**len(lnls)):  # Updated to use len(lnls)
        state_list[i] = [
            int(digit) for digit in change_base(i, 2, length=len(lnls))  # Updated length to len(lnls)
        ]
        
    ipsi_risks, contra_risks = get_risks_by_side(mean_risks, state_list, lnls)
    combined_risks = {f'ipsi {k}': v for k, v in ipsi_risks.items()}
    combined_risks.update({f'contra {k}': v for k, v in contra_risks.items()})
    ranked_combined = sorted(combined_risks.items(), key=lambda x: x[1])

    looper = 1
    treated_array = np.ones(len(ranked_combined))
    total_risk_new = 0
    sampled_total_risks_new = np.zeros(sampled_risks.shape[0])
    treated_array[:] = 1
    ipsi_idx = []
    contra_idx = []
    spared_lnls = []
    treated_lnls = ranked_combined.copy()
    while looper < len(lnls) * 2 + 2:
        # define which LNLs are treated
        if ci and (ci_single(sampled_total_risks_new)[1] >= threshold):
            spared_lnls = ranked_combined[:looper - 2]
            treated_lnls = ranked_combined[looper - 2:]
            break
        elif total_risk_new >= threshold:
            spared_lnls = ranked_combined[:looper - 2]
            treated_lnls = ranked_combined[looper - 2:]
            break
        total_risk = total_risk_new
        sampled_total_risk = sampled_total_risks_new
        treated_array[ipsi_idx] = 0
        treated_array[list(np.array(contra_idx) + 6)] = 0
        # exclude the next LNL from the target volume
        lnls_of_interest = [name for name, _ in ranked_combined[:looper]]
        ipsi_idx, contra_idx = get_lnl_indices(lnls_of_interest, lnls)
        idx_ipsi = get_state_indices(state_list, ipsi_idx)
        idx_contra = get_state_indices(state_list, contra_idx)
        not_idx_ipsi = np.setdiff1d(np.arange(state_list.shape[0]), idx_ipsi) #we get all the indices of the ipsilateral that are in the target volume

        # calculate risk of the spared LNLs
        # if no ipsi LNLs are excluded from the target volume, we simply sum the contra risks and vice versa
        if not ipsi_idx:
            total_risk_new = mean_risks.T[idx_contra].sum()
            sampled_total_risks_new = sampled_risks.transpose(0, 2, 1)[:, idx_contra].sum(axis=(1, 2))
        elif not contra_idx:
            total_risk_new = mean_risks[idx_ipsi].sum()
            sampled_total_risks_new = sampled_risks[:, idx_ipsi].sum(axis=(1, 2))
        else:
            total_risk_new = (
                mean_risks[idx_ipsi].sum() +
                mean_risks.T[idx_contra][:, not_idx_ipsi].sum()
            )
            sampled_total_risks_new = (
                sampled_risks[:, idx_ipsi].sum(axis=(1, 2)) +
                sampled_risks.transpose(0, 2, 1)[:, idx_contra][:, :, not_idx_ipsi].sum(axis=(1, 2))
            )
        looper += 1

    treated_ipsi = [name.split()[1] for name, _ in treated_lnls if name.startswith("ipsi")]
    treated_contra = [name.split()[1] for name, _ in treated_lnls if name.startswith("contra")]

    return (
        spared_lnls,
        total_risk,
        ranked_combined,
        treated_lnls,
        treated_array,
        treated_ipsi,
        treated_contra,
        sampled_total_risk,
    )


def analysis_treated_lnls_combinations_old(combinations, samples, model, threshold = 0.10):
    treatment_array = np.zeros((len(combinations),12))
    top3_spared = []
    diagnose_looper = {"ipsi": {'treatment_diagnose':{
        "I": 0,
        "II": 0,
        "III": 0,
        "IV": 0,
        "V": 0,
        "VII": 0
    }},
    "contra": {'treatment_diagnose':{
        "I": 0,
        "II": 0,
        "III": 0,
        "IV": 0,
        "V": 0,
        "VII": 0
    }}}
    treated_lnls_all = []
    treated_lnls_no_risk = []
    total_risks = np.zeros(len(combinations))
    sampled_risks_array = np.zeros((len(combinations),216))
    treated_ipsi_all = []
    treated_contra_all = []
    for index, pattern in enumerate(combinations):
        treated_looper = set()
        stage = pattern[0]
        midline_extension = pattern[1]
        counter_ipsi = 0
        for lnl_ipsi, status in diagnose_looper['ipsi']['treatment_diagnose'].items():
            diagnose_looper['ipsi']['treatment_diagnose'][lnl_ipsi] = pattern[2+counter_ipsi]
            counter_ipsi += 1
        counter_contra = 0
        for lnl_contra, status in diagnose_looper['contra']['treatment_diagnose'].items():
            diagnose_looper['contra']['treatment_diagnose'][lnl_contra] = pattern[8+counter_contra]
            counter_contra += 1
        sampled_risks, mean_risk = risk_sampled(samples = samples, model = model, t_stage = stage, given_diagnoses=diagnose_looper,midline_extension=midline_extension)     
        spared_lnls, total_risk, ranked_combined, treated_lnls, treated_array, treated_ipsi, treated_contra, sampled_total_risks =levels_to_spare_old(threshold, model, mean_risk, sampled_risks)
        for i in treated_lnls:
            treated_looper.add(i[0])
        treated_lnls_all.append(treated_lnls)
        treated_lnls_no_risk.append(treated_looper)
        treatment_array[index] = treated_array
        total_risks[index] = total_risk
        sampled_risks_array[index] = sampled_total_risks
        top3_spared.append(spared_lnls[::-1][:3])
        treated_ipsi_all.append(treated_ipsi)
        treated_contra_all.append(treated_contra)
    return treated_lnls_no_risk, treated_lnls_all, treatment_array, top3_spared, total_risks, treated_ipsi_all, treated_contra_all, sampled_risks_array


def analysis_treated_lnls_combinations(combinations, samples, model, threshold = 0.10, central = False, ci = True):
    pattern_index = 1 if central else 2
    treatment_array = np.zeros((len(combinations),12))
    top3_spared = []
    lnls_ranked =[]
    diagnose_looper = {"ipsi": {'treatment_diagnose':{
        "I": 0,
        "II": 0,
        "III": 0,
        "IV": 0,
        "V": 0,
        "VII": 0
    }},
    "contra": {'treatment_diagnose':{
        "I": 0,
        "II": 0,
        "III": 0,
        "IV": 0,
        "V": 0,
        "VII": 0
    }}}
    treated_lnls_all = []
    treated_lnls_no_risk = []
    cis = [[],[]]
    total_risks = np.zeros(len(combinations))
    sampled_risks_array = np.zeros((len(combinations),216))
    treated_ipsi_all = []
    treated_contra_all = []
    for index, pattern in enumerate(combinations):
        treated_looper = set()
        stage = pattern[0]
        midline_extension = pattern[1]
        counter_ipsi = 0
        for lnl_ipsi, status in diagnose_looper['ipsi']['treatment_diagnose'].items():
            diagnose_looper['ipsi']['treatment_diagnose'][lnl_ipsi] = pattern[pattern_index+counter_ipsi]
            counter_ipsi += 1
        counter_contra = 0
        for lnl_contra, status in diagnose_looper['contra']['treatment_diagnose'].items():
            diagnose_looper['contra']['treatment_diagnose'][lnl_contra] = pattern[pattern_index +6 +counter_contra]
            counter_contra += 1
        sampled_risks, mean_risk = risk_sampled(samples = samples, model = model, t_stage = stage, given_diagnoses=diagnose_looper,midline_extension=midline_extension, central = central)     
        spared_lnls, total_risk, ranked_combined, treated_lnls, treated_array, treated_ipsi, treated_contra, sampled_total_risks =levels_to_spare(threshold, model, mean_risk, sampled_risks, ci = True)
        for i in treated_lnls:
            treated_looper.add(i[0])
        treated_lnls_all.append(treated_lnls)
        treated_lnls_no_risk.append(treated_looper)
        treatment_array[index] = treated_array
        total_risks[index] = total_risk
        sampled_risks_array[index] = sampled_total_risks
        top3_spared.append(spared_lnls[::-1][:3])
        lnls_ranked.append(ranked_combined)  
        treated_ipsi_all.append(treated_ipsi)
        treated_contra_all.append(treated_contra)
        ci = ci_single(sampled_total_risks)
        cis[0].append(ci[0])
        cis[1].append(ci[1])
    return treated_lnls_no_risk, treated_lnls_all, treatment_array, top3_spared, total_risks, treated_ipsi_all, treated_contra_all, sampled_risks_array, lnls_ranked, cis


def analysis_treated_lnls_combinations_central_old(combinations,model, samples) :
    treatment_array = np.zeros((len(combinations),12))
    top3_spared = []
    diagnose_looper = {"ipsi": {'treatment_diagnose':{
        "I": 0,
        "II": 0,
        "III": 0,
        "IV": 0,
        "V": 0,
        "VII": 0
    }},
    "contra": {'treatment_diagnose':{
        "I": 0,
        "II": 0,
        "III": 0,
        "IV": 0,
        "V": 0,
        "VII": 0
    }}}
    treated_lnls_all = []
    treated_lnls_no_risk = []
    total_risks = np.zeros(len(combinations))
    sampled_risks_array = np.zeros((len(combinations),216))
    treated_ipsi_all = []
    treated_contra_all = []
    for index, pattern in enumerate(combinations):
        treated_looper = set()
        stage = pattern[0]
        counter_ipsi = 0
        for lnl_ipsi, status in diagnose_looper['ipsi']['max_llh_diagnose'].items():
            diagnose_looper['ipsi']['max_llh_diagnose'][lnl_ipsi] = pattern[1+counter_ipsi]
            counter_ipsi += 1
        counter_contra = 0
        for lnl_contra, status in diagnose_looper['contra']['max_llh_diagnose'].items():
            diagnose_looper['contra']['max_llh_diagnose'][lnl_contra] = pattern[7+counter_contra]
            counter_contra += 1
        sampled_risks, mean_risk = risk_sampled(samples = samples, model = model, t_stage = stage, given_diagnoses=diagnose_looper,central = True)     
        spared_lnls, total_risk, ranked_combined, treated_lnls, treated_array, treated_ipsi, treated_contra, sampled_total_risks =levels_to_spare_old(0.10, model, mean_risk, sampled_risks)
        for i in treated_lnls:
            treated_looper.add(i[0])
        treated_lnls_all.append(treated_lnls)
        treated_lnls_no_risk.append(treated_looper)
        treatment_array[index] = treated_array
        total_risks[index] = total_risk
        sampled_risks_array[index] = sampled_total_risks
        top3_spared.append(spared_lnls[::-1][:3])
        treated_ipsi_all.append(treated_ipsi)
        treated_contra_all.append(treated_contra)
    return treated_lnls_no_risk, treated_lnls_all, treatment_array, top3_spared, total_risks, treated_ipsi_all, treated_contra_all, sampled_risks_array



def count_number_treatments(treated_lnls_no_risk):
    """
    Function to calculate how many unique treatments are present.
    
    This function takes a list of sets, converts each set to an immutable `frozenset`,
    and counts how many times each unique frozenset appears in the list. The counts
    are stored in a dictionary where the keys are the frozensets and the values are
    their respective counts.

    Args:
        treated_lnls_no_risk (list of set): A list containing sets of treated lymph nodes
                                            or other elements.

    Returns:
        dict: A dictionary where the keys are frozensets representing unique sets from
              the input list, and the values are integers representing the count of
              occurrences for each unique frozenset.
    """
    set_counts = {}
    # Iterate through the list and update the counts in the dictionary
    for value in treated_lnls_no_risk:
        frozen_set = frozenset(value)  # Convert the set to a frozenset
        if frozen_set in set_counts:
            set_counts[frozen_set] += 1
        else:
            set_counts[frozen_set] = 1
    return set_counts