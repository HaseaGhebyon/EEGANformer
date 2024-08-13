from pathlib import Path

def get_config():
    return {
        "transformer_size" : 1,
        "batch_size" : 8,
        "num_epoch" : 200,
        "saving_epoch_step": 5,
        "learning_rate" : 2e-04,
        "epsilon" : 1e-9,

        "eeg_datasource" : "/content/drive/Shareddrives/Tugas Akhir/MI_database/3chan_5st_277dp",
        "image_datasource" : "/content/drive/Shareddrives/Tugas Akhir/ArrowIMG",
        "target_generation" : "/content/drive/Shareddrives/Tugas Akhir/results",
        "experiment_name": "runs/eegformermodel",
        "model_basename" : "eegformer_model",
        "model_folder" : "weights",
        "preload" : "latest",
        "num_cls" : 5,

        "channel_order" : ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'],
        "selected_channel" : ['C3', 'Cz', 'C4'],

        "bandpass_order" : 1,
        "seq_len" : 277,
        "low_cut" : 0.57,
        "high_cut" : 70.0,
        "samp_freq" : 200.0,
    }

def get_weights_file_path(config, epoch: str, type:str):
    assert type == "generator" or type == "discriminator"
    model_folder = f"{config['eeg_datasource']}_{type}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}_{epoch}"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config, type:str):
    assert type == "generator" or type == "discriminator"
    model_folder = f"{config['eeg_datasource']}_{type}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

def get_database_name(config, additional:str=None):
    chan_size = len(config["selected_channel"])
    seq_len = config["seq_len"]
    return f"{chan_size}chan_{seq_len}dp_{additional}"