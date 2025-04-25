import numpy as np


def sample_from_flattened(flattened_samples, num_samples=100, spaced=False):
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
        step_size = flattened_samples.shape[0] // num_samples
        return flattened_samples[::step_size][:num_samples]
    else:
        # Select random samples without replacement
        indices = np.random.choice(flattened_samples.shape[0], size=num_samples, replace=False)
        return flattened_samples[indices]


def risk_sampled(samples, model, t_stage, given_diagnoses, midline_extension):
    """
    Compute sampled risks and their mean for a given model and parameters.
    Note: The samples should already be thinned. use `sample_from_flattened` to thin the samples.

    Args:
        samples (list or np.ndarray): Sampled parameter sets.
        model (object): Model with a `risk` method.
        t_stage (str or int): Tumor stage.
        given_diagnoses (list or dict): Diagnoses information.
        midline_extension (bool): Indicates midline extension.

    Returns:
        tuple:
            - sampled_risks (np.ndarray): Computed risks for each sample.
            - mean_risk (np.ndarray): Mean risk across all samples.
    """
    sampled_risks = np.zeros(shape=(len(samples), model.risk().shape[0], model.risk().shape[1]), dtype=float)
    for i, sample in enumerate(samples):
        sampled_risks[i] = model.risk(given_params = sample, t_stage = t_stage, given_diagnoses = given_diagnoses,midline_extension=midline_extension) 
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

def levels_to_spare(threshold, model, risks, sampled_risks, ci=False):
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
    risks : numpy.ndarray
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
    state_list = model.noext.ipsi.state_list
    lnls = [lnl.name for lnl in model.noext.ipsi.lnls]

    ipsi_risks, contra_risks = get_risks_by_side(risks, state_list, lnls)
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
            total_risk_new = risks.T[idx_contra].sum()
            sampled_total_risks_new = sampled_risks.transpose(0, 2, 1)[:, idx_contra].sum(axis=(1, 2))
        elif not contra_idx:
            total_risk_new = risks[idx_ipsi].sum()
            sampled_total_risks_new = sampled_risks[:, idx_ipsi].sum(axis=(1, 2))
        else:
            total_risk_new = (
                risks[idx_ipsi].sum() +
                risks.T[idx_contra][:, not_idx_ipsi].sum()
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


def analysis_treated_lnls_sampled(combinations, samples, model) :
    """
    Analyzes treatment patterns and evaluates risks for given combinations and samples.

    This function processes a set of treatment combinations and samples to determine
    the treated lymph node levels (LNLs). To do so it uses the `levels_to_spare_ci` function
    which considers the upper bound of the CI.

    Args:
        combinations (list of tuples): A list of treatment patterns, where each pattern
            is a tuple containing stage, midline extension, and diagnostic statuses for
            ipsilateral and contralateral lymph node levels.
        samples (list): A list of sample data used for risk evaluation.

    Returns:
        tuple: A tuple containing the following elements:
            - treated_lnls_no_risk (list of sets): A list of sets, where each set contains
                treated lymph node levels without associated risks for each combination.
            - treated_lnls_all (list): A list of treated lymph node levels for all combinations.
            - treatment_array (numpy.ndarray): A 2D array where each row corresponds to a 
                combination and contains treatment-related metrics.
            - top3_spared (list): A list of the top 3 spared lymph node levels for each combination.
            - total_risks (numpy.ndarray): A 1D array containing the total risk for each combination.
            - treated_ipsi_all (list): A list of treated ipsilateral lymph node levels for all combinations.
            - treated_contra_all (list): A list of treated contralateral lymph node levels for all combinations.
            - sampled_risks_array (numpy.ndarray): A 2D array where each row corresponds to a combination
                and contains sampled risks for each sample.

    Notes:
        - The function modifies the `diagnose_looper` dictionary to update diagnostic
            statuses based on the input combinations.
    """
    treatment_array = np.zeros((len(combinations),12))
    top3_spared = []
    diagnose_looper = {'max_llh_diagnose':{
        "ipsi": {
            "I": 0,
            "II": 0,
            "III": 0,
            "IV": 0,
            "V": 0,
            "VII": 0,
        },
        "contra": {
            "I": 0,
            "II": 0,
            "III": 0,
            "IV": 0,
            "V": 0,
            "VII": 0,
        }
    }}
    treated_lnls_all = []
    treated_lnls_no_risk = []
    total_risks = np.zeros(len(combinations))
    sampled_risks_array = np.zeros((len(combinations),len(samples)))
    treated_ipsi_all = []
    treated_contra_all = []
    for index, pattern in enumerate(combinations):
        treated_looper = set()
        stage = pattern[0]
        midline_extension = pattern[1]
        counter_ipsi = 0
        for lnl_ipsi, status in diagnose_looper['max_llh_diagnose']['ipsi'].items():
            diagnose_looper['max_llh_diagnose']['ipsi'][lnl_ipsi] = pattern[2+counter_ipsi]
            counter_ipsi += 1
        counter_contra = 0
        for lnl_contra, status in diagnose_looper['max_llh_diagnose']['contra'].items():
            diagnose_looper['max_llh_diagnose']['contra'][lnl_contra] = pattern[8+counter_contra]
            counter_contra += 1
        sampled_risks, mean_risk = risk_sampled(samples = samples, model = model, t_stage = stage, given_diagnoses=diagnose_looper,midline_extension=midline_extension)  
        spared_lnls, total_risk, ranked_combined, treated_lnls, treated_array, treated_ipsi, treated_contra, sampled_total_risks = levels_to_spare_ci(0.10, model, mean_risk, sampled_risks)
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