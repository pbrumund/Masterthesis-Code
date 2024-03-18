#Code adapted from https://github.com/metno/NWPdocs/wiki/Examples

import numpy as np
import netCDF4
import pyproj
import datetime
import csv
import os.path

plot = False
save = True
# Location for wind forecast
lat = 61.21444
lon = 2.27222
# date range
start_time = datetime.datetime(2020,1,1)
end_time = datetime.datetime(2022,12,31)
# variables of interest
variables_1 = []   # MET post-processed
variables_2 = ["x_wind_10m", "y_wind_10m"]#, "atmosphere_convective_inhibition"]    # MEPS

filename_save = start_time.strftime("%Y%m%d") + "-" + end_time.strftime("%Y%m%d") + "_forecast_rerun1.csv"
end_time = end_time + datetime.timedelta(days=1)

headers = []
for v in ["times1"]+variables_1+["times2"]+variables_2:
    headers.extend([v+"_sh", v+"_mh", v+"_lh"])
if not os.path.exists(filename_save):
    with open(filename_save, "w", newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(headers)
else:   # load already downloaded predictions to find out last saved timestamp
    import fileloading
    table = fileloading.load_forecast(start_time, end_time-datetime.timedelta(days=1), filename=filename_save)
    last_pred_time = table['times2_sh'][-1]
    start_time = last_pred_time.astype(datetime.datetime) + datetime.timedelta(hours=1)
    del table
setup1 = True #only need to find grid index once
setup2 = True
n_errors = 0

time = start_time
while(time < end_time):
    print(time)

    year = time.year
    month = time.month
    day = time.day
    hour = time.hour

    # MET post-processed
    # try:
    #     filename = f"https://thredds.met.no/thredds/dodsC/metpparchive/{year}/{month:02}/{day:02}/met_forecast_1_0km_nordic_{year}{month:02}{day:02}T{hour:02}Z.nc"
    #     ncfile   = netCDF4.Dataset(filename,"r")

    #     if setup1:
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
    #         x = ncfile.variables["x"][:]
    #         y = ncfile.variables["y"][:]

    #         Ix1 = np.argmin(np.abs(x - X))
    #         Iy1 = np.argmin(np.abs(y - Y))

    #         setup1 = False
    #     predictions_i = {}
    #     predictions_i['times1'] = ncfile.variables["time"][:]
    #     for variable in variables_1:
    #         predictions_i[variable] = ncfile.variables[variable][:,Iy1,Ix1]
    # except:
    #     with open("missing_dates.txt", 'a') as errorfile:
    #         errorfile.write(time.strftime("%d.%m.%Y. %H:%M: post-processed data missing\n"))
    #         print('Post-processed data missing')
    #     for key in variables_1 + ["times1"]:
    #         predictions_i[key] = predictions_i[key][6:]
            # n_errors += 1

    # MEPS
    meps_filenames = [
        f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{year}/{month:02}/{day:02}/meps_det_2_5km_{year}{month:02}{day:02}T{hour:02}Z.nc",
        f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{year}/{month:02}/{day:02}/meps_subset_2_5km_{year}{month:02}{day:02}T{hour:02}Z.nc",
        f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{year}/{month:02}/{day:02}/meps_mbr0_full_2_5km_{year}{month:02}{day:02}T{hour:02}Z.nc",
        f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{year}/{month:02}/{day:02}/meps_mbr1_full_2_5km_{year}{month:02}{day:02}T{hour:02}Z.nc",
        f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{year}/{month:02}/{day:02}/meps_lagged_6_h_subset_2_5km_{year}{month:02}{day:02}T{hour:02}Z.nc"
    ]
    errors = 0
    for filename in meps_filenames:
        try:
            #filename = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{year}/{month:02}/{day:02}/meps_mbr0_full_2_5km_{year}{month:02}{day:02}T{hour:02}Z.nc"
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
            predictions2_i = {}
            predictions2_i['times2'] = ncfile.variables["time"][:]
            for variable in variables_2:
                if "subset" in filename:
                    predictions2_i[variable] = ncfile.variables[variable][:,0,0,Iy2,Ix2]
                else:
                    predictions2_i[variable] = ncfile.variables[variable][:,0,Iy2,Ix2]
            n_errors = 0
            predictions2_i_last = predictions2_i
            break
        except:
            errors += 1
            if errors == len(meps_filenames):
                with open("missing_dates.txt", 'a') as errorfile:
                    errorfile.write(time.strftime("%d.%m.%Y. %H:%M: MEPS data missing\n"))
                    print('MEPS data missing')
                for key in variables_2 + ["times2"]:
                    predictions2_i_last[key] = predictions2_i_last[key][6:]
                n_errors += 1
    # predictions_i.update(predictions2_i)
    predictions_i = predictions2_i_last
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
    if n_errors >= 5:
        raise RuntimeError('Connection unsuccessfull')
    time += datetime.timedelta(hours=6)
    