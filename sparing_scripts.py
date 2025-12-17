import numpy as np
import lymph

def sample_from_flattened(flattened_samples, num_samples=100, spaced=False, step_size = None):
    """
    Sample from flattened MCMC samples, either randomly or evenly spaced.

    Args:
        flattened_samples: Flattened MCMC samples (shape: [num_samples, num_parameters]).
        num_samples: Number of samples to select (default: 100).
        spaced: If True, select evenly spaced samples; if False, random (default: False).
        step_size: Step size for spaced sampling (default: auto-calculated).

    Returns:
        Selected samples as numpy array.
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
    """
    Convert integer to another base (2-16).

    Args:
        number: Integer to convert (must be non-negative).
        base: Target base (2-16).
        reverse: If True, reverse digit order (default: False).
        length: Minimum length, pads with zeros if needed (default: auto).

    Returns:
        String representation of converted number.
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
    Compute posterior state distribution for each parameter sample.

    Args:
        samples: Parameter samples, each matching model.get_params() structure.
        model: Lymph model (Unilateral or Midline).
        t_stage: T-stage ('early' or 'late').
        midline_extension: Midline extension status (Midline models only).
        given_diagnoses: Diagnosis dict with ipsi/contra structure.
        central: Whether tumor is central (Midline models only).

    Returns:
        tuple: (sampled_risks, mean_risk) - posterior state distributions per sample and their mean.
    """
    sampled_risks = np.zeros((len(samples), *model.posterior_state_dist().shape), dtype=float)
    for i, sample in enumerate(samples):
        params = {key: sample[i] for i, key in enumerate(model.get_params().keys())}
        model.set_params(**params)
        if type(model) == lymph.models.unilateral.Unilateral:
            sampled_risks[i] = model.posterior_state_dist(t_stage = t_stage, given_diagnosis = given_diagnoses) 
        else:
            sampled_risks[i] = model.posterior_state_dist(t_stage = t_stage, given_diagnosis = given_diagnoses, midext = midline_extension, central = central) 
    mean_risk = sampled_risks.mean(axis = 0)
    return sampled_risks, mean_risk


def ci_single(sampled_risks, level=0.95):
    """
    Calculate credibility interval for sampled risks.

    Args:
        sampled_risks: Array of sampled risks.
        level: Credibility level (default: 0.95).

    Returns:
        Array with [lower, upper] bounds.
    """
    lower = (1 - level) / 2 * 100
    upper = 100 - lower
    ci = np.percentile(sampled_risks, [lower, upper])
    return ci


def ci_multiple(sampled_risks_set, level=0.95):
    """
    Calculate credibility intervals for multiple risk sets.

    Args:
        sampled_risks_set: Array of sampled risks for multiple cases.
        level: Credibility level (default: 0.95).

    Returns:
        Array with shape (n_cases, 2) containing [lower, upper] bounds per case.
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


def sparing_bilateral(threshold, model, mean_risks, sampled_risks, ci=False):
    """
    Determine LNL sparing for Midline models.
    
    Internal function called by levels_to_spare for Midline models.
    
    Args:
        threshold: Maximum risk threshold.
        model: lymph.models.Midline instance.
        mean_risks: Mean posterior state distribution.
        sampled_risks: Sampled posterior state distributions.
        ci: Use CI upper bound for threshold (default: False).
    
    Returns:
        tuple: (spared_lnls, total_risk, ranked_combined, treated_lnls, treated_array,
                treated_ipsi, treated_contra, sampled_total_risk)
    """
    lnls = list(model.noext.ipsi.graph.lnls.keys())
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
        treated_array[list(np.array(contra_idx) + len(lnls))] = 0
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
        if ci:
            spared_lnls = ranked_combined[:looper - 2]
            treated_lnls = ranked_combined[looper - 2:]
        else:
            spared_lnls = ranked_combined[:looper - 2]
            treated_lnls = ranked_combined[looper - 2:]

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
    
def sparing_unilateral(threshold, model, mean_risks, sampled_risks, ci=False):
    """
    Determine LNL sparing for unilateral models.
    
    Internal function called by levels_to_spare for Unilateral models.
    
    Args:
        threshold: Maximum risk threshold.
        model: lymph.models.Unilateral instance.
        mean_risks: Mean posterior state distribution.
        sampled_risks: Sampled posterior state distributions.
        ci: Use CI upper bound for threshold (default: False).
    
    Returns:
        tuple: (spared_lnls, total_risk, ranked, treated_lnls, treated_lnls_names,
                treated_array, sampled_total_risk)
    """
    lnls = list(model.graph.lnls.keys())
    state_list = np.zeros((2**len(lnls), len(lnls)))
    for i in range(2**len(lnls)):
        state_list[i] = [
            int(digit) for digit in change_base(i, 2, length=len(lnls))
        ]
        
    state_list = np.zeros((2**len(lnls), len(lnls)))
    for i in range(2**len(lnls)):
        state_list[i] = [
            int(digit) for digit in change_base(i, 2, length=len(lnls))
        ]
        
    risks = {lnl: mean_risks[state_list[:, i] == 1].sum() for i, lnl in enumerate(lnls)}
    ranked = sorted(risks.items(), key=lambda x: x[1])

    looper = 1
    treated_array = np.ones(len(ranked))
    total_risk_new = 0
    sampled_total_risks_new = np.zeros(sampled_risks.shape[0])
    treated_array[:] = 1
    idx_lnls_to_treat = []
    spared_lnls = []
    treated_lnls = ranked.copy()
    while looper < len(lnls) + 2:
        # define which LNLs are treated
        if ci and (ci_single(sampled_total_risks_new)[1] >= threshold):
            spared_lnls = ranked[:looper - 2]
            treated_lnls = ranked[looper - 2:]
            break
        elif total_risk_new >= threshold:
            spared_lnls = ranked[:looper - 2]
            treated_lnls = ranked[looper - 2:]
            break
        total_risk = total_risk_new
        sampled_total_risk = sampled_total_risks_new
        treated_array[idx_lnls_to_treat] = 0
        # exclude the next LNL from the target volume
        lnls_of_interest = [name for name, _ in ranked[:looper]]
        idx_lnls_to_treat = [lnls.index(name) for name in lnls_of_interest]
        idx = get_state_indices(state_list, idx_lnls_to_treat)

        # calculate risk of the spared LNLs

        total_risk_new = mean_risks[idx].sum()
           
        sampled_total_risks_new = sampled_risks[:, idx].sum(axis = 1)
        looper += 1
        if ci:
            spared_lnls = ranked[:looper - 2]
            treated_lnls = ranked[looper - 2:]
        else:
            spared_lnls = ranked[:looper - 2]
            treated_lnls = ranked[looper - 2:]

        treated_lnls_names = [name for name, _ in treated_lnls]
    return (
        spared_lnls,
        total_risk,
        ranked,
        treated_lnls,
        treated_lnls_names,
        treated_array,
        sampled_total_risk,
    )   


def levels_to_spare(threshold, model, mean_risks, sampled_risks, ci=False):
    """
    Determine which LNLs can be spared while keeping total risk below threshold.
    
    Ranks LNLs by risk and iteratively excludes lowest-risk LNLs from treatment
    until total risk of untreated regions exceeds threshold.
    
    Args:
        threshold: Maximum allowable total risk (e.g., 0.10 for 10%).
        model: Lymph model (Unilateral or Midline).
        mean_risks: Mean posterior state distribution.
        sampled_risks: Sampled posterior state distributions for uncertainty.
        ci: If True, use CI upper bound for threshold; if False, use mean (default: False).
    
    Returns:
        tuple: (spared_lnls, total_risk, ranked_lnls, treated_lnls, treated_array, 
                treated_ipsi*, treated_contra*, sampled_total_risk)
        *Only for Midline models
    """
    if threshold <= 0:
        raise ValueError("Threshold must be larger than zero")
    if isinstance(model, lymph.models.Midline):
        return sparing_bilateral(threshold, model, mean_risks, sampled_risks, ci)
    elif isinstance(model, lymph.models.unilateral.Unilateral):
        return sparing_unilateral(threshold, model, mean_risks, sampled_risks, ci)
    else:
        raise TypeError("Model must be an instance of lymph.models.Midline or lymph.models.unilateral.Unilateral")



def analysis_treated_lnls_combinations_bilateral(combinations, samples, model, threshold = 0.10, central = False, ci = True):
    """
    Analyze treatment recommendations for multiple diagnostic combinations (bilateral/Midline models).
    
    Processes diagnostic combinations (T-stage, midline extension, LNL patterns) and determines
    optimal treatment for each using levels_to_spare.
    
    Args:
        combinations: List of tuples with (t_stage, midline_ext, *lnl_pattern).
                     For central tumors: 13 elements (no midline_ext); otherwise 14.
        samples: MCMC parameter samples.
        model: lymph.models.Midline instance.
        threshold: Risk threshold for treatment decisions (default: 0.10).
        central: If True, tumor is central (default: False).
        ci: If True, use CI for threshold comparison (default: True).
    
    Returns:
        tuple: (treated_lnls_no_risk, treated_lnls_all, treatment_array, top3_spared,
                total_risks, treated_ipsi_all, treated_contra_all, sampled_risks_array,
                lnls_ranked, cis)
    """
    if isinstance(model, lymph.models.Midline):
        lnls = list(model.noext.ipsi.graph.lnls.keys())
    else:
        raise TypeError("Model must be an instance of lymph.models.Midline")
    pattern_index = 1 if central else 2
    treatment_array = np.zeros((len(combinations),len(lnls)*2))
    top3_spared = []
    lnls_ranked =[]
    lnls = list(model.noext.ipsi.graph.lnls.keys())
    diagnose_looper = {"ipsi":{'treatment_diagnose':{}}, 
                      "contra":{'treatment_diagnose':{}}}
    for lnl in lnls:
        diagnose_looper['ipsi']['treatment_diagnose'][lnl] = 0
        diagnose_looper['contra']['treatment_diagnose'][lnl] = 0
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
            diagnose_looper['contra']['treatment_diagnose'][lnl_contra] = pattern[pattern_index +len(lnls) +counter_contra]
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


def analysis_treated_lnls_combinations_unilateral(combinations, samples, model, threshold = 0.10, ci = True):
    """
    Analyze treatment recommendations for multiple diagnostic combinations (unilateral models).
    
    Args:
        combinations: List of tuples with (t_stage, *lnl_pattern).
        samples: MCMC parameter samples.
        model: lymph.models.Unilateral instance.
        threshold: Risk threshold (default: 0.10).
        ci: If True, use CI for threshold comparison (default: True).
    
    Returns:
        tuple: (treated_lnls_no_risk, treated_lnls_all, treatment_array, top3_spared,
                total_risks, sampled_risks_array, lnls_ranked, cis)
    """
    if isinstance(model, lymph.models.unilateral.Unilateral):
        lnls = list(model.graph.lnls.keys())
    else:
        raise TypeError("Model must be an instance of lymph.models.unilateral.Unilateral")
    pattern_index = 1
    treatment_array = np.zeros((len(combinations),len(lnls)))
    top3_spared = []
    lnls_ranked =[]
    lnls = list(model.graph.lnls.keys())
    diagnose_looper = {'treatment_diagnose':{}}
    for lnl in lnls:
        diagnose_looper['treatment_diagnose'][lnl] = 0
    treated_lnls_all = []
    treated_lnls_no_risk = []
    cis = [[],[]]
    total_risks = np.zeros(len(combinations))
    sampled_risks_array = np.zeros((len(combinations),len(samples)))
    treated_all = []
    for index, pattern in enumerate(combinations):
        treated_looper = set()
        stage = pattern[0]
        counter = 0
        for lnl, status in diagnose_looper['treatment_diagnose'].items():
            diagnose_looper['treatment_diagnose'][lnl] = pattern[pattern_index+counter]
            counter += 1
        sampled_risks, mean_risk = risk_sampled(samples = samples, model = model, t_stage = stage, given_diagnoses=diagnose_looper)     
        spared_lnls, total_risk, ranked_combined, treated_lnls, treated_lnls_names, treated_array, sampled_total_risks =levels_to_spare(threshold, model, mean_risk, sampled_risks, ci = True)
        for i in treated_lnls:
            treated_looper.add(i[0])
        treated_lnls_all.append(treated_lnls)
        treated_lnls_no_risk.append(treated_looper)
        treatment_array[index] = treated_array
        total_risks[index] = total_risk
        sampled_risks_array[index] = sampled_total_risks
        top3_spared.append(spared_lnls[::-1][:3])
        lnls_ranked.append(ranked_combined)  
        ci = ci_single(sampled_total_risks)
        cis[0].append(ci[0])
        cis[1].append(ci[1])
    return treated_lnls_no_risk, treated_lnls_all, treatment_array, top3_spared, total_risks, sampled_risks_array, lnls_ranked, cis


def count_number_treatments(treated_lnls_no_risk):
    """
    Count occurrences of unique treatment combinations.

    Args:
        treated_lnls_no_risk: List of sets containing treated LNL names.

    Returns:
        dict: Frozensets (unique treatments) mapped to occurrence counts.
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