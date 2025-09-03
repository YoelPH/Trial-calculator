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
        model (lymph.models.Midline): A lymph model object with `set_params` and `risk` methods.
        t_stage (str): The T stage input to be passed to the model's risk calculation (e.g., 'early', 'late').
        midline_extension (bool, optional): Additional parameter for midline extension risk calculation, default is None.
        given_diagnoses (dict, optional): Diagnosis information for risk calculation with ipsi/contra structure, default is None.
        central (bool, optional): Whether to use central tumor calculation, default is False.

    Returns:
        tuple: 
            - sampled_risks (numpy.ndarray): Array of risk values computed for each sample, shape (num_samples, number of LNLs**2, number of LNLs**2).
            - mean_risk (numpy.ndarray): Mean of the sampled risks across all samples, shape (number of LNLs**2, number of LNLs**2).
    """
    sampled_risks = np.zeros((len(samples), *model.risk().shape), dtype=float)
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
    """
    Calculate per-LNL risks for ipsilateral and contralateral sides.
    
    Args:
        risks (numpy.ndarray): Risk matrix with shape (num_states, num_states).
        state_list (numpy.ndarray): Binary state list indicating LNL involvement.
        lnls (list): List of lymph node level names.
        
    Returns:
        tuple: (ipsi_risks, contra_risks) - dictionaries mapping LNL names to risk values.
    """
    ipsi = {lnl: risks[state_list[:, i] == 1].sum() for i, lnl in enumerate(lnls)}
    contra = {lnl: risks.T[state_list[:, i] == 1].sum() for i, lnl in enumerate(lnls)}
    return ipsi, contra

def get_lnl_indices(lnl_names, lnls):
    """
    Extract indices for ipsilateral and contralateral LNLs from named list.
    
    Args:
        lnl_names (list): List of LNL names with format ['ipsi I', 'contra II', ...].
        lnls (list): List of base LNL names ['I', 'II', ...].
        
    Returns:
        tuple: (ipsi_indices, contra_indices) - lists of indices for each side.
    """
    ipsi, contra = [], []
    for name in lnl_names:
        side, lnl = name.split()
        (ipsi if side == 'ipsi' else contra).append(lnls.index(lnl))
    return ipsi, contra

def get_state_indices(state_list, indices):
    """
    Get unique state indices where any of the specified LNLs are involved.
    
    Args:
        state_list (numpy.ndarray): Binary state list indicating LNL involvement.
        indices (list): List of LNL indices to check.
        
    Returns:
        numpy.ndarray: Unique state indices where specified LNLs are involved.
    """
    combined = []
    for idx in indices:
        combined.extend(np.where(state_list[:, idx] == 1)[0])
    return np.unique(combined)


def levels_to_spare(threshold, model, mean_risks, sampled_risks, ci=False):
    """
    Determine the levels of lymph nodes to spare based on a risk threshold.
    
    This function evaluates the risks associated with ipsilateral (ipsi) and 
    contralateral (contra) lymph node levels (LNLs) and determines which levels 
    can be spared while keeping the total risk below a specified threshold.
    
    The algorithm works by:
    1. Ranking all LNLs by individual risk (lowest to highest)
    2. Iteratively excluding LNLs from treatment starting with lowest risk
    3. Checking if total risk exceeds threshold (using mean risk or CI upper bound)
    4. Stopping when threshold is exceeded and returning the previous configuration
    
    Parameters:
    -----------
    threshold : float
        The maximum allowable total risk for sparing lymph node levels (e.g., 0.10 for 10%).
    model : lymph.models.Midline
        A lymph Midline model object containing the LNL structure and parameters.
    mean_risks : numpy.ndarray
        Array of mean risks associated with each state in the model, shape (num_states, num_states).
    sampled_risks : numpy.ndarray
        Array of sampled risks for uncertainty estimation, with shape 
        (num_samples, num_states, num_states).
    ci : bool, optional
        If True, uses confidence interval upper bound for threshold comparison.
        If False, uses mean risk for threshold comparison. Default is False.
    
    Returns:
    --------
    spared_lnls : list of tuples
        List of spared lymph node levels and their associated risks, sorted by risk.
    total_risk : float
        The total risk corresponding to the returned treatment decision.
    ranked_combined : list of tuples
        List of all lymph node levels and their associated risks, sorted by risk.
    treated_lnls : list of tuples
        List of treated lymph node levels and their associated risks.
    treated_array : numpy.ndarray
        Array indicating which LNLs are treated (0 for treated, 1 for spared).
    treated_ipsi : list of str
        Names of treated ipsilateral lymph node levels.
    treated_contra : list of str
        Names of treated contralateral lymph node levels.
    sampled_total_risk : numpy.ndarray
        Array of sampled total risks corresponding to the returned treatment decision.
    
    Notes:
    ------
    - The function dynamically determines LNL structure from the model
    - Risk and treatment decision are consistent (unlike the deprecated old version)
    - The function handles both CI-based and mean risk-based thresholding
    """
    if threshold <= 0:
        raise ValueError("Threshold must be larger than zero")
    if isinstance(model, lymph.models.Midline):
        lnls = list(model.noext.ipsi.graph.lnls.keys())
    else:
        raise TypeError("Model must be an instance of lymph.models.Midline")

    state_list = np.zeros((2**len(lnls), len(lnls)))
    for i in range(2**len(lnls)):
        state_list[i] = [
            int(digit) for digit in change_base(i, 2, length=len(lnls))
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
    

def analysis_treated_lnls_combinations(combinations, samples, model, threshold = 0.10, central = False, ci = True):
    """
    Analyze treatment recommendations for multiple diagnostic combinations.
    
    This function processes a set of diagnostic combinations (T-stage, midline extension,
    and LNL involvement patterns) and determines the optimal treatment strategy for each
    using the levels_to_spare function.
    
    Args:
        combinations (list): List of diagnostic combination tuples. Each tuple contains:
            - T-stage (str): 'early' or 'late'
            - Midline extension (bool): True/False (only for non-central tumors)
            - LNL involvement pattern (bool): 12 boolean values for ipsi/contra LNL involvement
        samples (array-like): MCMC parameter samples for uncertainty quantification.
        model (lymph.models.Midline): Lymph model object.
        threshold (float, optional): Risk threshold for treatment decisions. Default is 0.10 (10%).
        central (bool, optional): Whether to analyze central tumors (affects combination indexing). Default is False.
        ci (bool, optional): Whether to use confidence intervals for threshold decisions. Default is True.
    
    Returns:
        tuple: Contains the following arrays/lists for all combinations:
            - treated_lnls_no_risk (list): List of sets containing treated LNL names (without risk values)
            - treated_lnls_all (list): List of full treated LNL information with risk values
            - treatment_array (numpy.ndarray): Binary array indicating treatment decisions (shape: num_combinations x 12)
            - top3_spared (list): List of top 3 spared LNLs for each combination
            - total_risks (numpy.ndarray): Array of total risks for each combination
            - treated_ipsi_all (list): List of treated ipsilateral LNL names for each combination
            - treated_contra_all (list): List of treated contralateral LNL names for each combination
            - sampled_risks_array (numpy.ndarray): Array of sampled risks for each combination
            - lnls_ranked (list): List of LNL rankings by risk for each combination
            - cis (list): List of confidence intervals [lower_bounds, upper_bounds]
    
    Notes:
        - For central tumors, combinations exclude midline extension (13 elements vs 14)
        - The function uses multiprocessing-friendly design for large combination sets
        - All outputs are aligned by combination index for easy analysis
    """
    if isinstance(model, lymph.models.Midline):
        lnls = list(model.noext.ipsi.graph.lnls.keys())
    else:
        raise TypeError("Model must be an instance of lymph.models.Midline")
    pattern_index = 1 if central else 2
    treatment_array = np.zeros((len(combinations),len(lnls)*2))
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
    sampled_risks_array = np.zeros((len(combinations),len(samples)))
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