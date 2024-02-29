import datetime

def get_gp_opt(**kwargs):
    opt = {'t_start': datetime.datetime(2020,1,1),
           't_end': datetime.datetime(2022,12,31,23,50),
           'start_date_train': datetime.datetime(2020,1,1),
           'end_date_train': datetime.datetime(2021,12,31),
           'n_z': 100,
           'epochs_first_training': 250,
           'max_epochs_second_training': 10,
           'epochs_timeseries_first_train': 500,
           'epochs_timeseries_retrain': 500,
           'loss_lb': 0.5,
           'verbose': True,
           'n_last': 36,   # 12 h
           'train_posterior': True,
           'steps_forward': 60,
           't_start_score': datetime.datetime(2022,1,1),
           't_end_score': datetime.datetime(2022,12,31),
           'freq_retrain': 36,
           'reselect_data': True,
           'cashing': True,
           'dt_meas': 10,
           'dt_pred': 10,
           'direct_model_order': 3,
           'iterative_model_order': 5,
           'n_samples_mc': 100
           }
    opt.update(kwargs)
    return opt
    # TODO: 500 epochs