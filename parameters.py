glodbal_settings = {
  "agents": ['Nico'], # Eric, Nico, Sanne (birds in this set)
  "params": ['longitude', 'latitude'],
  "network": 'transformer', # rnn, transformer
  "len_input": 1, # size of input vectors
  "prediction_window": 0, # how many steps in future do you want to "jump"
  "len_pred": 3800, # number of points to forcast
  "epochs": 100,
  "batch_size": 32,
  "data_path": '../saved_data/dataset_bird.csv',
  "model_path_rnn": '../saved_model/RNN_model_',
  "model_path_transformer": '../saved_model/transformer_model_',
  "scaler_path": '../saved_scaler/scaler.sav'
}