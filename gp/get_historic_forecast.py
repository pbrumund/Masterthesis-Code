#Code adapted from https://github.com/metno/NWPdocs/wiki/Examples

import numpy as np
import netCDF4
import pyproj
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import datetime

plot = False
save = True
# Location for wind forecast
lat = 61.21444
lon = 2.27222
# date range
start_time = datetime.datetime(2020,1,1)
end_time = datetime.datetime(2022,12,31)

filename_save = start_time.strftime("%Y%m%d") + "-" + end_time.strftime("%Y%m%d") + ".csv"
end_time = end_time + datetime.timedelta(days=1)

setup = True #only need to find grid index once
nerrors = 0

times_1 = np.ma.array([]) #forecast for next 6 hours
times_2 = np.ma.array([])  #forecast between 6 and 12 hours
times_3 = np.ma.array([])  #forecast between 12 and 18 hours
wind_speeds_1 = np.ma.array([])
wind_speeds_2 = np.ma.array([])
wind_speeds_3= np.ma.array([])

time = start_time
while(time < end_time):
    print(time)

    year = time.year
    month = time.month
    day = time.day
    hour = time.hour

    filename = f"https://thredds.met.no/thredds/dodsC/metpparchive/{year}/{month:02}/{day:02}/met_forecast_1_0km_nordic_{year}{month:02}{day:02}T{hour:02}Z.nc"
    try:
        ncfile   = netCDF4.Dataset(filename,"r")
        if setup:
            # Get indices of closest grid point to coordinates on first iteration
            # Code adapted from https://github.com/metno/NWPdocs/wiki/Examples
            crs = pyproj.CRS.from_cf(
                {
                    "grid_mapping_name": "lambert_conformal_conic",
                    "standard_parallel": [63.3, 63.3],
                    "longitude_of_central_meridian": 15.0,
                    "latitude_of_projection_origin": 63.3,
                    "earth_radius": 6371000.0,
                }
            )
            # Transformer to project from ESPG:4368 (WGS:84) to our lambert_conformal_conic
        
            proj = pyproj.Proj.from_crs(4326, crs, always_xy=True)
            # Compute projected coordinates of lat/lon point

            X,Y = proj.transform(lon,lat)

            # Find nearest neighbour
            x = ncfile.variables["x"][:]
            y = ncfile.variables["y"][:]

            Ix = np.argmin(np.abs(x - X))
            Iy = np.argmin(np.abs(y - Y))
            setup = False

        times_i        = ncfile.variables["time"][:]
        wind_speeds_i = ncfile.variables["wind_speed_10m"][:,Iy,Ix]
        nerrors = 0
    except:
        # missing data on server, use last prediction
        with open("missing_dates.txt", 'a') as errorfile:
            errorfile.write(time.strftime("%d.%m.%Y. %H:%M\n"))
        nerrors += 1
        times_i = times_i[6:]
        wind_speeds_i = wind_speeds_i[6:]       
    finally:
        if save:
            with open(filename_save, 'a') as file:
                t1, t2, t3 = times_i[:6], times_i[6:12], times_i[12:18]
                w1, w2, w3 = wind_speeds_i[:6], wind_speeds_i[6:12], wind_speeds_i[12:18]
                np.savetxt(file, np.array([np.ma.resize(t1,6),np.ma.resize(w1,6),np.ma.resize(t2,6),np.ma.resize(w2,6),np.ma.resize(t3,6),np.ma.resize(w3,6)]).T, delimiter=';', fmt='%s')
            times_1 = np.ma.concatenate([times_1, times_i[:6]])
        if plot:
            wind_speeds_1 = np.ma.concatenate([wind_speeds_1, wind_speeds_i[:6]])
            times_2 = np.ma.concatenate([times_2, times_i[6:12]])
            wind_speeds_2 = np.ma.concatenate([wind_speeds_2, wind_speeds_i[6:12]])
            times_3 = np.ma.concatenate([times_3, times_i[12:18]])
            wind_speeds_3 = np.ma.concatenate([wind_speeds_3, wind_speeds_i[12:18]])
        time = time + datetime.timedelta(hours=6)
if plot:
    times1_plt = [datetime.datetime.fromtimestamp(time) for time in times_1]
    times2_plt = [datetime.datetime.fromtimestamp(time) for time in times_2]
    times3_plt = [datetime.datetime.fromtimestamp(time) for time in times_3]
    plt.plot(times1_plt,wind_speeds_1, color='g')
    plt.plot(times2_plt,wind_speeds_2, color='r')
    plt.plot(times3_plt,wind_speeds_3, color='b')
    plt.show()