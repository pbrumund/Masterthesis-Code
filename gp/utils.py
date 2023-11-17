import numpy as np
import datetime

def get_NWP(wind_table, time, steps=0, key="wind_speed_10m"):
    """Interpolates weather prediction from dict using current forecast for the given time
    
    Uses linear interpolation to get values between full hours
    Keyword Arguments:
    wind_table -- dictionary of NWP and measurements
    time -- time at current time step
    steps -- number of steps to predict ahead
    key -- string with name of value as in dict, standard: "wind_speed_10m"
    """
    r = time.hour % 6
    if key in ["wind_speed_10m", "wind_direction_10m", "air_pressure_at_sea_level", "air_temperature_2m"]:   # MET post-processed
        times_sh = wind_table["times1_sh"]
    else:
        times_sh = wind_table['times2_sh']
    dt = time - times_sh[0].astype(datetime.datetime)
    i_start = int(dt.total_seconds()//21600) * 6 #6 hours, time of last released forecast
    predicted_trajectory = np.concatenate([wind_table[key+"_sh"][i_start:i_start+6], 
                                           wind_table[key+"_mh"][i_start:i_start+6], 
                                           wind_table[key+"_lh"][i_start:]])    # Reconstruct the most recent NWP at the given time
    predicted_times = np.concatenate([wind_table["times1_sh"][i_start:i_start+6], 
                                           wind_table["times1_mh"][i_start:i_start+6], 
                                           wind_table["times1_lh"][i_start:]])    # Reconstruct the most recent NWP at the given time
    t_rounded = time.replace(minute=0)

    i = np.where(predicted_times==(time+datetime.timedelta(minutes=10*steps)).replace(minute=0))[0][0]
    # i = int((6*r + time.minute//10 + steps)//6)
    wind_interp = (predicted_trajectory[i+1]*(time+steps*datetime.timedelta(minutes=10)).minute/60 
                   + predicted_trajectory[i]*(1-(time+steps*datetime.timedelta(minutes=10)).minute/60))
    return wind_interp

def get_wind_value(wind_table, time, steps=0):
    """Returns measured wind speed, inputs as in get_NWP"""
    dt = time - wind_table['times_meas'][0]
    i = int(dt.total_seconds()/600) + steps
    return wind_table['meas'][i]