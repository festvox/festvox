# Default hyperparameters:
hparams = {}

# Audio:
hparams['num_mels']=80
hparams['num_freq']=1025
hparams['sample_rate']=16000
hparams['frame_length_ms']=50
hparams['frame_shift_ms']=12.5
hparams['preemphasis']=0.97
hparams['min_level_db']=-100
hparams['ref_level_db']=20

# Model:
hparams['outputs_per_step']=5
hparams['padding_idx']=None
hparams['use_memory_mask']=True

# Data loader
hparams['pin_memory']=False
hparams['num_workers']=4

# Training:
hparams['batch_size']=32
hparams['adam_beta1']=0.9
hparams['adam_beta2']=0.999
hparams['initial_learning_rate']=0.002
hparams['decay_learning_rate']=True
hparams['nepochs']=20000
hparams['weight_decay']=0.0
hparams['clip_thresh']=1.0

# Save
hparams['checkpoint_interval']=10000
hparams['save_states_interval']=1000

# Eval:
hparams['max_iters']=200
hparams['griffin_lim_iters']=60
hparams['power']=1.5              # Power to raise magnitudes to prior to Griffin-Lim

# Enhancements
hparams['exponential_moving_average']=None
hparams['ema_decay']=0.9999

# Vocoder
hparams['highpass_cutoff']=70.0
hparams['global_gain_scale']=1.0 



class hyperparameters(dict):

   def __init__(self):

      for (k,v) in hparams.items():
          self.__setattr__(k,v)

   def __setattr__(self, name, value):
        self[name] = value

   def update_params(self, read_arr):
      for line in read_arr:
          line = line.split('\n')[0]
          #print(line)
          key, val = line.split('=')[0], line.split('=')[1]
          self.__setattr__(key,val)

   def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)
