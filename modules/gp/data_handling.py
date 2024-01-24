import numpy as np
import datetime

from .fileloading import load_weather_data

class DataHandler():
    """Put functions from utils into a class to avoid passing data every time"""
    def __init__(self, t_start, t_end, opt) -> None:
        self.weather_data = load_weather_data(t_start, t_end)
        self.dt_meas = opt['dt_meas']   # interval of measurements

    def get_time_since_forecast(self, time, steps):
        t_start_predict = time.replace(hour=time.hour//6*6)
        i_start = np.where(self.weather_data['times1_sh']==t_start_predict)
        t_predict = time+datetime.timedelta(minutes=self.dt_meas*steps)
        time_since_pred = t_predict-t_start_predict
        return time_since_pred.total_seconds()/3600

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
    
    def generate_features(self, time, n_last=3, feature='error', steps_ahead = 0):
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
        # Next hour
        if '3 steps' in feature:
            # cape_next = self.get_NWP(time, steps_ahead+1, 'specific_convective_available_potential_energy')
            # sqrt_cape_next = np.sqrt(cape_next)
            # temperature_next = self.get_NWP(time, steps_ahead+1, 'air_temperature_2m')
            # humidity_next = self.get_NWP(time, steps_ahead+1, 'relative_humidity_2m')
            # wind_speed_of_gust_next = self.get_NWP(time, steps_ahead+1, 'wind_speed_of_gust_diff')
            wind_prediction_at_step_next = self.get_NWP(time, steps_ahead+1, 'wind_speed_10m')
            # pressure_next = self.get_NWP(time, steps_ahead+1, 'air_pressure_at_sea_level')
            # # Normalize inputs
            # sqrt_cape_next = ((sqrt_cape_next-self.weather_data['means']['sqrt_specific_convective_available_potential_energy_sh'])
            #     /self.weather_data['std']['sqrt_specific_convective_available_potential_energy_sh'])
            # temperature_next = ((temperature_next-self.weather_data['means']['air_temperature_2m_sh'])
            #             /self.weather_data['std']['air_temperature_2m_sh'])
            # humidity_next = ((humidity_next-self.weather_data['means']['relative_humidity_2m_sh'])
            #             /self.weather_data['std']['relative_humidity_2m_sh'])
            # pressure_next = ((pressure_next-self.weather_data['means']['air_pressure_at_sea_level_sh'])
            #             /self.weather_data['std']['air_pressure_at_sea_level_sh'])
            wind_prediction_at_step_next = ((wind_prediction_at_step_next - self.weather_data['means']['wind_speed_10m_sh'])
                                    /self.weather_data['std']['wind_speed_10m_sh'])
            # wind_speed_of_gust_next = ((wind_speed_of_gust_next - self.weather_data['means']['wind_speed_of_gust_diff_sh'])
            #                     /self.weather_data['std']['wind_speed_of_gust_diff_sh'])
            time_previous = time - datetime.timedelta(hours=1)
            # cape_previous = self.get_NWP(time_previous, steps_ahead, 'specific_convective_available_potential_energy')
            # sqrt_cape_previous = np.sqrt(cape_previous)
            # temperature_previous = self.get_NWP(time_previous, steps_ahead, 'air_temperature_2m')
            # humidity_previous = self.get_NWP(time_previous, steps_ahead, 'relative_humidity_2m')
            # wind_speed_of_gust_previous = self.get_NWP(time_previous, steps_ahead, 'wind_speed_of_gust_diff')
            wind_prediction_at_step_previous = self.get_NWP(time_previous, steps_ahead, 'wind_speed_10m')
            # pressure_previous = self.get_NWP(time_previous, steps_ahead, 'air_pressure_at_sea_level')
            # # Normalize inputs
            # sqrt_cape_previous = ((sqrt_cape_previous-self.weather_data['means']['sqrt_specific_convective_available_potential_energy_sh'])
            #     /self.weather_data['std']['sqrt_specific_convective_available_potential_energy_sh'])
            # temperature_previous = ((temperature_previous-self.weather_data['means']['air_temperature_2m_sh'])
            #             /self.weather_data['std']['air_temperature_2m_sh'])
            # humidity_previous = ((humidity_previous-self.weather_data['means']['relative_humidity_2m_sh'])
            #             /self.weather_data['std']['relative_humidity_2m_sh'])
            # pressure_previous = ((pressure_previous-self.weather_data['means']['air_pressure_at_sea_level_sh'])
            #             /self.weather_data['std']['air_pressure_at_sea_level_sh'])
            wind_prediction_at_step_previous = ((wind_prediction_at_step_previous - self.weather_data['means']['wind_speed_10m_sh'])
                                    /self.weather_data['std']['wind_speed_10m_sh'])
            # wind_speed_of_gust_previous = ((wind_speed_of_gust_previous - self.weather_data['means']['wind_speed_of_gust_diff_sh'])
            #                     /self.weather_data['std']['wind_speed_of_gust_diff_sh'])
            NWP_values_previous_next = [wind_prediction_at_step_previous, wind_prediction_at_step_next] 
                # wind_speed_of_gust_previous, wind_speed_of_gust_next, sqrt_cape_previous, sqrt_cape_next,
                # temperature_previous, temperature_next, humidity_previous, humidity_next, 
                # pressure_previous-pressure_next]
        if feature == 'nwp': return np.array(NWP_values)
        elif feature == 'nwp 3 steps': return np.array(NWP_values+NWP_values_previous_next)
        elif feature == 'nwp & time':
            # add time in days
            dt = time+steps_ahead*datetime.timedelta(minutes=self.dt_meas)-self.weather_data['times_meas'][0]
            t_out = dt.total_seconds()/(60*60*24)
            time_since_pred = self.get_time_since_forecast(time, steps_ahead)
            return np.concatenate([NWP_values, [time_since_pred, t_out]])
        elif feature == 'nwp 3 steps & time':
            # add time in days
            dt = time+steps_ahead*datetime.timedelta(minutes=self.dt_meas)-self.weather_data['times_meas'][0]
            t_out = dt.total_seconds()/(60*60*24)
            time_since_pred = self.get_time_since_forecast(time, steps_ahead)
            return np.concatenate([NWP_values+NWP_values_previous_next, [time_since_pred, t_out]])
        # TODO: Add time since forecast release, sample from medium/long horizon as well
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
        elif feature == 'error & nwp':
            return np.concatenate([errors, NWP_values])
        elif feature == 'error & nwp 3 steps':
            return np.concatenate([errors, NWP_values, NWP_values_previous_next])
        elif feature == 'error & nwp & time':
            # add time in days
            dt = time+steps_ahead*datetime.timedelta(minutes=self.dt_meas)-self.weather_data['times_meas'][0]
            t_out = dt.total_seconds()/(60*60*24)
            time_since_pred = self.get_time_since_forecast(time, steps_ahead)
            return np.concatenate([errors, np.append(NWP_values + [time_since_pred], t_out)])
        elif feature == 'error & nwp 3 steps & time':
            # add time in days
            dt = time+steps_ahead*datetime.timedelta(minutes=self.dt_meas)-self.weather_data['times_meas'][0]
            t_out = dt.total_seconds()/(60*60*24)
            time_since_pred = self.get_time_since_forecast(time, steps_ahead)
            return np.concatenate([errors, np.append(NWP_values+NWP_values_previous_next+time_since_pred, t_out)])
        elif feature == 'measurement & nwp':
            return np.concatenate([measurements, NWP_values])
        elif feature == 'measurement & nwp & time':
            # add time in days
            dt = time+steps_ahead*datetime.timedelta(minutes=self.dt_meas)-self.weather_data['times_meas'][0]
            t_out = dt.total_seconds()/(60*60*24)
            time_since_pred = self.get_time_since_forecast(time, steps_ahead)
            return np.concatenate([measurements, np.append(NWP_values + [time_since_pred], t_out)])
        elif feature == 'measurement & nwp 3 steps & time':
            # add time in days
            dt = time+steps_ahead*datetime.timedelta(minutes=self.dt_meas)-self.weather_data['times_meas'][0]
            t_out = dt.total_seconds()/(60*60*24)
            time_since_pred = self.get_time_since_forecast(time, steps_ahead)
            return np.concatenate([measurements, np.append(NWP_values+NWP_values_previous_next+[time_since_pred], t_out)])
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