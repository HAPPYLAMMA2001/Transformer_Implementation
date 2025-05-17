from pathlib import Path

def get_config():
    return {
        "batch_size": 64,
        "epochs": 25,
        "lr": 1e-4,
        "seq_len": 260,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "ur",
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        'local_dataset_path': "train.json"
    }

def get_weights_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/ model_folder/model_filename)