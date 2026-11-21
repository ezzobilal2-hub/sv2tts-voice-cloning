
## Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3
mel_n_channels = 40
encoder_sampling_rate = 16000     # Sampling rate of encoder

mel_window_length = 25.0          # Window length in ms
mel_window_step = 10.0            # Step/hop size in ms

## Training parameters
learning_rate_init = 1e-4
speakers_per_batch = 64
utterances_per_speaker = 10
