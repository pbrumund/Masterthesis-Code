import datetime

def get_gp_opt(**kwargs):
    opt = {'t_start': datetime.datetime(2020,1,1),
           't_end': datetime.datetime(2022,12,31,23,50),
           'start_date_train': datetime.datetime(2020,1,1),
           'end_date_train': datetime.datetime(2021,12,31),
           'n_z': 50,
           'epochs_first_training': 100,
           'max_epochs_second_training': 50,
           'epochs_timeseries_first_train': 100,
           'epochs_timeseries_retrain': 10,
           'loss_lb': 0.5,
           'verbose': True,
           'n_last': 432,   # 3 days
           'train_posterior': True,
           'steps_forward': 100,
           't_start_score': datetime.datetime(2022,1,1),
           't_end_score': datetime.datetime(2022,12,31),
           'freq_retrain': 36,
           'reselect_data': True
           }
    opt.update(kwargs)
    return opt
    