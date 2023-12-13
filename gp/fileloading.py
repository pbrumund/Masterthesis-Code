import numpy as np
import datetime
import csv

def load_forecast(start_time, end_time, filename=None):
    """Loads numerical weather predictions from a csv file
    
    Keyword arguments:
    start_time -- start of the time window for predictions, first part of file name
    end_time -- end of the time window for predictions, second part of file name
    Returns times and predictions for time windows of 0-6, 6-12 and 12-18 h after release of newest weather forecast
    """
    if filename is None:
        filename = ('gp\\weather_data\\' + start_time.strftime("%Y%m%d") + "-" 
                    + end_time.strftime("%Y%m%d") + "_forecast.csv")
    prediction_dict = {}
    with open(filename, mode="r") as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)
        means_vec = next(reader)
        std_vec = next(reader)
        means = {}
        std = {}
        for i, var in enumerate(headers):
            means[var] = float(means_vec[i])
            std[var] = float(std_vec[i])
        table = np.array(list(reader))
        for i, h in enumerate(headers):
            column_i = table[:,i]
            if "times" in h:
                column_i = np.asarray(column_i.astype(float), dtype='datetime64[s]')
            else:
                column_i = column_i.astype(float)
            prediction_dict[h] = column_i
    prediction_dict['means'] = means
    prediction_dict['std'] = std
    return prediction_dict



def load_weather_measurements(start_time, end_time, filename=None):
    """Loads wind measurements from a csv file
    
    Keyword arguments:
    start_time -- start of the time window for predictions, first part of file name
    end_time -- end of the time window for predictions, second part of file name
    Returns times and measurements
    """
    if filename is None:
        filename = ('gp\\weather_data\\' + start_time.strftime("%Y%m%d") + "-" 
                    + end_time.strftime("%Y%m%d") + "_observations.csv")
    with open(filename, mode="r") as f:
        reader = csv.reader(f, delimiter=';')
        table = np.array(list(reader))
        times = table[:,0].astype(np.datetime64)
        # times_sh = np.asarray(times_sh, dtype='datetime64[s]')
        wind_data = table[:,1].astype(float)
    return times, wind_data

def load_weather_data(start_time, end_time, filename=None):
    """Loads NWP and measurements, returns them as a dictionary"""
    times_obs, wind_speed_obs = load_weather_measurements(start_time, end_time, filename)
    predictions = load_forecast(start_time, end_time, filename)
    wind_speeds_forecast = np.array([])
    for key in predictions:
        if key in ["wind_speed_10m_sh", "wind_speed_10m_mh", "wind_speed_10m_lh"]:
            wind_speeds_forecast = np.append(wind_speeds_forecast, predictions[key])
    c_correction = np.mean(wind_speed_obs)/np.mean(wind_speeds_forecast)
    for key in predictions:
        if key in ["wind_speed_10m_sh", "wind_speed_10m_mh", "wind_speed_10m_lh"]:
            predictions[key] *= c_correction
    # times_sh, wind_forecast_sh, times_mh, wind_forecast_mh, times_lh, wind_forecast_lh = load_prediction(start_time, end_time)
    # c_correction = np.mean(wind_speed_obs)/np.mean(wind_forecast_sh)
    times_interp = np.array([times_obs[0]+i*np.timedelta64(10, 'm') 
        for i in range(int((times_obs[-1].astype(float)-times_obs[0].astype(float))//600000))])
    times_interp_datetime = np.array(
        [times_obs[0].astype(datetime.datetime)+i*datetime.timedelta(minutes=10) 
            for i in range(int((times_obs[-1].astype(float)-times_obs[0].astype(float))//600000))])
    
    wind_obs_interp = np.interp((times_interp.astype(float)-times_interp[0].astype(float))/
                        (times_interp[-1].astype(float)-times_interp[0].astype(float)), 
                        (times_obs.astype(float)-times_obs[0].astype(float))/
                        (times_obs[-1].astype(float)-times_obs[0].astype(float)), wind_speed_obs)   # Interpolate to fill missing values
    data = {'times_meas': times_interp_datetime,
            'meas': wind_obs_interp}
    data.update(predictions)
    return data