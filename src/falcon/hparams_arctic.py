import tensorflow as tf


hparams_wavenet = tf.contrib.training.HParams(

    # Training:
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    amsgrad=False,
    initial_learning_rate=1e-3,
    # see lrschedule.py for available lr_schedule
    lr_schedule="noam_learning_rate_decay",
    lr_schedule_kwargs={},  # {"anneal_rate": 0.5, "anneal_interval": 50000},
    nepochs=2000,
    weight_decay=0.0,
    clip_thresh=-1,
    # max time steps can either be specified as sec or steps
    # if both are None, then full audio samples are used in a batch
    max_time_sec=None,
    max_time_steps=8000,
    # Hold moving averaged parameters and use them for evaluation
    exponential_moving_average=True,
    # averaged = decay * averaged + (1 - decay) * x
    ema_decay=0.9999,

)

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
    cleaners='english_cleaners',
    use_cmudict=False,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

    # Audio:
    num_mels=80,
    num_freq=1025,
    sample_rate=16000,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,

    # Model:
    # TODO: add more configurable hparams
    outputs_per_step=5,
    padding_idx=None,
    use_memory_mask=False,

    # Data loader
    pin_memory=True,
    num_workers=2,

    # Training:
    batch_size=32,
    adam_beta1=0.9,
    adam_beta2=0.999,
    initial_learning_rate=0.002,
    decay_learning_rate=True,
    nepochs=1000,
    weight_decay=0.0,
    clip_thresh=1.0,

    # Save
    checkpoint_interval=1000,

    # Eval:
    max_iters=200,
    griffin_lim_iters=60,
    power=1.5,              # Power to raise magnitudes to prior to Griffin-Lim
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
