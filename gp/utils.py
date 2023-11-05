import numpy as np
import datetime

def get_NWP(wind_table, time, steps):
    """Interpolates weather prediction from dict using current forecast for the given time
    
    Uses linear interpolation to get values between full hours
    Keyword Arguments:
    wind_table -- dictionary of NWP and measurements
    time -- time at current time step
    steps -- number of steps to predict ahead
    """
    r = time.hour % 6
    dt = time - wind_table['times_sh'][0].astype(datetime.datetime)
    i_start = int(dt.total_seconds()//21600) * 6 #6 hours, time of last released forecast
    predicted_trajectory = np.concatenate([wind_table['pred_sh'][i_start:i_start+6], 
                                           wind_table['pred_mh'][i_start:i_start+6], 
                                           wind_table['pred_lh'][i_start:]])    # Reconstruct the most recent NWP at the given time
    
    i = (6*r + time.minute//10 + steps)//6
    wind_interp = (predicted_trajectory[i+1]*(time+steps*datetime.timedelta(minutes=10)).minute/60 
                   + predicted_trajectory[i]*(1-(time+steps*datetime.timedelta(minutes=10)).minute/60))
    return wind_interp

def get_wind_value(wind_table, time, steps):
    """Returns measured wind speed, inputs as in get_NWP"""
    dt = time - wind_table['times_meas'][0]
    i = int(dt.total_seconds()/600) + steps
    return wind_table['meas'][i]