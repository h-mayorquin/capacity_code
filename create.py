import pandas as pd

data_dic = {'hypercolumns': -1, 'minicolumns':-1, 'sequence_length':-1, 'capacity':-1,  'p_critical':-1, 'trials':-1}

data_frame = pd.DataFrame(data_dic, index={0})

data_frame.to_csv('../storage_capacity_data.csv')
