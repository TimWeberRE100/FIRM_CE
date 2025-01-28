######## RECEIVE A PROFILE - DO FOR EACH NODE
def get_frequency_profile(timeseries_profile):
    return frequency_profile

def get_normalised_profile(timeseries_profile):
    frequency_profile = get_frequency_profile(timeseries_profile)

    return normalised_frequency_profile

def get_dc_offset(frequency_profile):
    return dc_offset

#### CUTOFFS FOR EACH NODE
def get_bandpass_filter(lower_cutoff, upper_cutoff):
    return bandpass_profile

def get_filtered_frequency(frequency_profile, bandpass_filter_profile):
    return filtered_frequency_profile

def get_timeseries_profile(frequency_profile):
    return timeseries_profile

def apportion_dc_offset(dc_offset, timeseries_profiles):
    return timeseries_profiles_with_dc

######## Pass flexible profile

######## Pass storage power profile (negative for charging)

######## Pass stored energy profile