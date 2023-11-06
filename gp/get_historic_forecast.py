#Code adapted from https://github.com/metno/NWPdocs/wiki/Examples

import numpy as np
import netCDF4
import pyproj
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import datetime
import csv

plot = False
save = True
# Location for wind forecast
lat = 61.21444
lon = 2.27222
# date range
start_time = datetime.datetime(2020,1,1)
end_time = datetime.datetime(2020,1,31)
# variables of interest
variables_1 = ["wind_speed_10m", "wind_direction_10m", "air_pressure_at_sea_level", "air_temperature_2m"]   # MET post-processed
variables_2 = ["specific_convective_available_potential_energy", "atmosphere_convective_inhibition"]    # MEPS

filename_save = start_time.strftime("%Y%m%d") + "-" + end_time.strftime("%Y%m%d") + "_forecast.csv"
end_time = end_time + datetime.timedelta(days=1)

headers = []
for v in ["times1"]+variables_1+["times2"]+variables_2:
    headers.extend([v+"_sh", v+"_mh", v+"_lh"])
with open(filename_save, "w", newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(headers)
setup1 = True #only need to find grid index once
setup2 = True
# nerrors = 0

# times_1 = np.ma.array([]) #forecast for next 6 hours
# times_2 = np.ma.array([])  #forecast between 6 and 12 hours
# times_3 = np.ma.array([])  #forecast between 12 and 18 hours
# wind_speeds_1 = np.ma.array([])
# wind_speeds_2 = np.ma.array([])
# wind_speeds_3= np.ma.array([])

time = start_time
while(time < end_time):
    print(time)

    year = time.year
    month = time.month
    day = time.day
    hour = time.hour

    # MET post-processed
    try:
        filename = f"https://thredds.met.no/thredds/dodsC/metpparchive/{year}/{month:02}/{day:02}/met_forecast_1_0km_nordic_{year}{month:02}{day:02}T{hour:02}Z.nc"
        ncfile   = netCDF4.Dataset(filename,"r")

        if setup1:
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

            Ix1 = np.argmin(np.abs(x - X))
            Iy1 = np.argmin(np.abs(y - Y))

            setup1 = False
        predictions_i = {}
        predictions_i['times1'] = ncfile.variables["time"][:]
        for variable in variables_1:
            predictions_i[variable] = ncfile.variables[variable][:,Iy1,Ix1]
    except:
        with open("missing_dates.txt", 'a') as errorfile:
            errorfile.write(time.strftime("%d.%m.%Y. %H:%M: post-processed data missing\n"))
            print('Post-processed data missing')
        for key in variables_1 + ["times1"]:
            predictions_i[key] = predictions_i[key][6:]

    # MEPS
    try:
        filename = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{year}/{month:02}/{day:02}/meps_mbr0_full_2_5km_{year}{month:02}{day:02}T{hour:02}Z.nc"
        ncfile   = netCDF4.Dataset(filename,"r")

        if setup2:
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

            Ix2 = np.argmin(np.abs(x - X))
            Iy2 = np.argmin(np.abs(y - Y))

            setup2 = False
        predictions_i['times2'] = ncfile.variables["time"][:]
        for variable in variables_2:
            predictions_i[variable] = ncfile.variables[variable][:,0,Iy2,Ix2]
    except:
        try:
            filename = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{year}/{month:02}/{day:02}/meps_mbr1_full_2_5km_{year}{month:02}{day:02}T{hour:02}Z.nc"
            ncfile   = netCDF4.Dataset(filename,"r")
            predictions_i['times2'] = ncfile.variables["time"][:]
            for variable in variables_2:
                predictions_i[variable] = ncfile.variables[variable][:,0,Iy2,Ix2]
        except:
            try:
                filename = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{year}/{month:02}/{day:02}/meps_lagged_6_h_subset_2_5km_{year}{month:02}{day:02}T{hour:02}Z.nc"
                ncfile   = netCDF4.Dataset(filename,"r")
                predictions_i['times2'] = ncfile.variables["time"][:]
                for variable in variables_2:
                    predictions_i[variable] = ncfile.variables[variable][:,0,Iy2,Ix2]
            except:
                with open("missing_dates.txt", 'a') as errorfile:
                    errorfile.write(time.strftime("%d.%m.%Y. %H:%M: MEPS data missing\n"))
                    print('MEPS data missing')
                for key in variables_1 + ["times1"]:
                    predictions_i[key] = predictions_i[key][6:]
    finally:
        if save:
            with open(filename_save, 'a') as file:
                v_list = []
                for key in predictions_i:
                    v = predictions_i[key]
                    v1, v2, v3 = v[:6], v[6:12], v[12:18]
                    v1, v2, v3 = np.ma.resize(v1,6), np.ma.resize(v2,6), np.ma.resize(v3,6)
                    v_list.extend([v1,v2,v3])
                v_array = np.array(v_list).T
                np.savetxt(file, v_array, delimiter=';', fmt='%s')
        time += datetime.timedelta(hours=6)

    # filename1 = f"https://thredds.met.no/thredds/dodsC/metpparchive/{year}/{month:02}/{day:02}/met_forecast_1_0km_nordic_{year}{month:02}{day:02}T{hour:02}Z.nc"
    # filename2 = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{year}/{month:02}/{day:02}/meps_mbr0_full_2_5km_{year}{month:02}{day:02}T{hour:02}Z.nc"
    # try:
    #     ncfile1   = netCDF4.Dataset(filename1,"r")
    #     ncfile2  = netCDF4.Dataset(filename2,"r")
    #     if setup:
    #         # Get indices of closest grid point to coordinates on first iteration
    #         # Code adapted from https://github.com/metno/NWPdocs/wiki/Examples
    #         crs = pyproj.CRS.from_cf(
    #             {
    #                 "grid_mapping_name": "lambert_conformal_conic",
    #                 "standard_parallel": [63.3, 63.3],
    #                 "longitude_of_central_meridian": 15.0,
    #                 "latitude_of_projection_origin": 63.3,
    #                 "earth_radius": 6371000.0,
    #             }
    #         )
    #         # Transformer to project from ESPG:4368 (WGS:84) to our lambert_conformal_conic
        
    #         proj = pyproj.Proj.from_crs(4326, crs, always_xy=True)
    #         # Compute projected coordinates of lat/lon point

    #         X,Y = proj.transform(lon,lat)

    #         # Find nearest neighbour
    #         x1 = ncfile1.variables["x"][:]
    #         y1 = ncfile1.variables["y"][:]

    #         Ix1 = np.argmin(np.abs(x1 - X))
    #         Iy1 = np.argmin(np.abs(y1 - Y))

    #         x2 = ncfile2.variables["x"][:]
    #         y2 = ncfile2.variables["y"][:]

    #         Ix2 = np.argmin(np.abs(x2 - X))
    #         Iy2 = np.argmin(np.abs(y2 - Y))
    #         setup = False

    #     predictions_i = {}
    #     predictions_i['times1'] = ncfile1.variables["time"][:]
    #     for variable in variables_1:
    #         predictions_i[variable] = ncfile1.variables[variable][:,Iy1,Ix1]
    #     predictions_i['times2'] = ncfile2.variables["time"][:]
    #     for variable in variables_2:
    #         predictions_i[variable] = ncfile2.variables[variable][:,0,Iy2,Ix2]
    #     # wind_speeds_i = ncfile.variables["wind_speed_10m"][:,Iy,Ix]
    #     # times2_i = ncfile2.variables["time"][:]
    #     # cape_i = ncfile2.variables["specific_convective_available_potential_energy"][:,Iy2,Ix2]
    #     nerrors = 0
    # except:
    #     # missing data on server, use last prediction
    #     with open("missing_dates.txt", 'a') as errorfile:
    #         errorfile.write(time.strftime("%d.%m.%Y. %H:%M\n"))
    #     nerrors += 1
    #     for key in predictions_i:
    #         predictions_i[key] = predictions_i[key][6:]
    #     print("missing data")
    #     # times_i = times_i[6:]
    #     # wind_speeds_i = wind_speeds_i[6:]
    #     # times2_i = times2_i[6:]
    #     # cape_i = cape_i[6:]       
    # finally:
    #     if save:
    #         with open(filename_save, 'a') as file:
    #             v_list = []
    #             for key in predictions_i:
    #                 v = predictions_i[key]
    #                 v1, v2, v3 = v[:6], v[6:12], v[12:18]
    #                 v1, v2, v3 = np.ma.resize(v1,6), np.ma.resize(v2,6), np.ma.resize(v3,6)
    #                 v_list.extend([v1,v2,v3])
    #             v_array = np.array(v_list).T
    #             np.savetxt(file, v_array, delimiter=';', fmt='%s')
    #     time = time + datetime.timedelta(hours=6)
                # t1, t2, t3 = times_i[:6], times_i[6:12], times_i[12:18]
                # w1, w2, w3 = wind_speeds_i[:6], wind_speeds_i[6:12], wind_speeds_i[12:18]
                # np.savetxt(file, np.array([np.ma.resize(t1,6),np.ma.resize(w1,6),np.ma.resize(t2,6),np.ma.resize(w2,6),np.ma.resize(t3,6),np.ma.resize(w3,6)]).T, delimiter=';', fmt='%s')
#             times_1 = np.ma.concatenate([times_1, times_i[:6]])
#         if plot:
#             wind_speeds_1 = np.ma.concatenate([wind_speeds_1, wind_speeds_i[:6]])
#             times_2 = np.ma.concatenate([times_2, times_i[6:12]])
#             wind_speeds_2 = np.ma.concatenate([wind_speeds_2, wind_speeds_i[6:12]])
#             times_3 = np.ma.concatenate([times_3, times_i[12:18]])
#             wind_speeds_3 = np.ma.concatenate([wind_speeds_3, wind_speeds_i[12:18]])
        
# if plot:
#     times1_plt = [datetime.datetime.fromtimestamp(time) for time in times_1]
#     times2_plt = [datetime.datetime.fromtimestamp(time) for time in times_2]
#     times3_plt = [datetime.datetime.fromtimestamp(time) for time in times_3]
#     plt.plot(times1_plt,wind_speeds_1, color='g')
#     plt.plot(times2_plt,wind_speeds_2, color='r')
#     plt.plot(times3_plt,wind_speeds_3, color='b')
#     plt.show()