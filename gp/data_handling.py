import numpy as np
import datetime
import csv
import random

from fileloading import load_weather_data

class DataHandler():
    def __init__(self, t_start, t_end, opt) -> None:
        self.weather_data = load_weather_data(t_start, t_end)
        self.dt_meas = opt['dt_meas']   # interval of measurements

    def get_NWP(self, time, steps=0, key="wind_speed_10m"):
        """Interpolates weather prediction from dict using current forecast for the given time
        
        Uses linear interpolation to get values between full hours
        Keyword Arguments:
        time -- time at current time step
        steps -- number of steps to predict ahead, >=0
        key -- string with name of value as in dict, standard: "wind_speed_10m"
        """
        if key in ["wind_speed_10m", "wind_direction_10m", "air_pressure_at_sea_level", "air_temperature_2m"]:   # MET post-processed
            times_sh = self.weather_data["times1_sh"]
        else:
            times_sh = self.weather_data['times2_sh']
        dt = time - times_sh[0].astype(datetime.datetime)
        i_start = int(dt.total_seconds()//21600) * 6 #6 hours, time of last released forecast
        predicted_trajectory = np.concatenate([self.weather_data[key+"_sh"][i_start:i_start+6], 
                                               self.weather_data[key+"_mh"][i_start:i_start+6], 
                                               self.weather_data[key+"_lh"][i_start:]])    # Reconstruct the most recent NWP at the given time
        predicted_times = np.concatenate([self.weather_data["times1_sh"][i_start:i_start+6], 
                                          self.weather_data["times1_mh"][i_start:i_start+6], 
                                          self.weather_data["times1_lh"][i_start:]])    # Reconstruct the most recent NWP at the given time
        i = np.where(predicted_times==(time+datetime.timedelta(minutes=self.dt_meas*steps)).replace(minute=0))[0][0]
        # interpolate between weather predictions between full hours
        nwp_interp = (predicted_trajectory[i+1]*(time+steps*datetime.timedelta(minutes=self.dt_meas)).minute/60 
                    + predicted_trajectory[i]*(1-(time+steps*datetime.timedelta(minutes=self.dt_meas)).minute/60))
        return nwp_interp
    
    def get_measurement(self, time, steps=0):
        """Returns measured wind speed, inputs as in get_NWP"""
        if time.minute%self.dt_meas != 0:
            # round down time to last measurement time, add difference to steps
            steps += time.minute%self.dt_meas/self.dt_meas
            time = time.replace(minute=time.minute//self.dt_meas*self.dt_meas)
        if steps%1 != 0:
            # interpolate linearly between measurements if time is between measurement times
            a = steps%1
            return (a*self.get_measurement(time, np.ceil(steps))
                    + (1-a)*self.get_measurement(time, np.floor(steps)))
        steps = int(steps)
        dt = time - self.weather_data['times_meas'][0]
        i = int(dt.total_seconds()/(60*self.dt_meas)) + steps
        return self.weather_data['meas'][i]
    
    def generate_features(self, time, n_last=3, feature='error', steps_ahead = 1):
        # Get NWP inputs
        cape = self.get_NWP(time, steps_ahead, 'specific_convective_available_potential_energy')
        sqrt_cape = np.sqrt(cape)
        temperature = self.get_NWP(time, steps_ahead, 'air_temperature_2m')
        humidity = self.get_NWP(time, steps_ahead, 'relative_humidity_2m')
        wind_speed_of_gust = self.get_NWP(time, steps_ahead, 'wind_speed_of_gust_diff')
        wind_prediction_at_step = self.get_NWP(time, steps_ahead, 'wind_speed_10m')
        pressure = self.get_NWP(time, steps_ahead, 'air_pressure_at_sea_level')
        # Normalize inputs
        sqrt_cape = ((sqrt_cape-self.weather_data['means']['sqrt_specific_convective_available_potential_energy_sh'])
            /self.weather_data['std']['sqrt_specific_convective_available_potential_energy_sh'])
        temperature = ((temperature-self.weather_data['means']['air_temperature_2m_sh'])
                    /self.weather_data['std']['air_temperature_2m_sh'])
        humidity = ((humidity-self.weather_data['means']['relative_humidity_2m_sh'])
                    /self.weather_data['std']['relative_humidity_2m_sh'])
        pressure = ((pressure-self.weather_data['means']['air_pressure_at_sea_level_sh'])
                    /self.weather_data['std']['air_pressure_at_sea_level_sh'])
        wind_prediction_at_step = ((wind_prediction_at_step - self.weather_data['means']['wind_speed_10m_sh'])
                                /self.weather_data['std']['wind_speed_10m_sh'])
        wind_speed_of_gust = ((wind_speed_of_gust - self.weather_data['means']['wind_speed_of_gust_diff_sh'])
                            /self.weather_data['std']['wind_speed_of_gust_diff_sh'])
        NWP_values = [
            wind_prediction_at_step, wind_speed_of_gust, sqrt_cape, temperature, humidity, pressure
            ]

        if feature == 'nwp': return np.array(NWP_values)
        elif feature == 'nwp & time':
            # add time in days
            dt = time+steps_ahead*datetime.timedelta(minutes=self.dt_meas)-self.weather_data['times_meas'][0]
            t_out = dt.total_seconds()/(60*60*24)
            return np.concatenate([NWP_values, [t_out]])
        
        # Only autoregressive models
        measurements = np.zeros(n_last)
        wind_predictions = np.zeros(n_last)

        start_time = time - datetime.timedelta(minutes=(n_last-1)*self.dt_meas)
        for i in range(n_last):
            measurements[i] = self.get_measurement(start_time, i)
            wind_predictions[i] = self.get_NWP(start_time, i, 'wind_speed_10m')

        errors = measurements - wind_predictions
        
        if feature == 'error':
            return errors
        elif feature == 'measurement':
            return measurements
        elif feature == 'measurement & error':
            return np.concatenate([measurements, errors])
        elif feature == 'prediction & error':
            return np.concatenate([wind_predictions, errors])
        elif feature == 'prediction & measurement':
            return np.concatenate([measurements, wind_prediction_at_step])
        elif feature == 'error & nwp':
            return np.concatenate([errors, [measurements[-1]], NWP_values])
        elif feature == 'error & nwp & time':
            return np.concatenate([errors, [measurements[-1]], NWP_values])
        elif feature == 'measurement & nwp':
            return np.concatenate([measurements, NWP_values])
        elif feature == 'measurement & nwp & time':
            return np.concatenate([measurements, NWP_values])
        else:
            raise ValueError('Unknown value for input feature')
        
    def generate_labels(self, time, label='error', steps_ahead = 1):
        measurement = self.get_measurement(time, steps=steps_ahead)
        if label == 'error':
            prediction = self.get_NWP(time, steps=steps_ahead)
            return measurement - prediction
        elif label == 'measurement':
            return measurement
        elif label == 'change of wind speed':
            current_measurement = self.get_measurement(time, steps=0)
            return measurement-current_measurement
        else:
            raise ValueError("Unknown value for feature")