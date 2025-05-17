import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

from dataset import CustomDataset, causal_mask
from config import get_weights_path, get_config
from model import build_transformer

from pathlib import Path
import json



def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]

def build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
              min_frequency=2
            )
        
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):

    with open(config['local_dataset_path'], 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Convert dict to list
    ds_raw = []
    for k, v in raw_data.items():
        ds_raw.append(v)
        
    


    tokenizer_src = build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 90 - 10 split
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = CustomDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = CustomDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0
    for item in train_ds:
        src_ids = tokenizer_src.encode(item['src_text']).ids
        # print("Source Text:", item['src_text'])
        # print("Tokenized Source IDs:", src_ids)
        tgt_ids = tokenizer_tgt.encode(item['tgt_text']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print("Max src:", max_len_src)
    print("Max tgt:", max_len_tgt)

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config,vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len, 
        vocab_tgt_len, 
        config['seq_len'], 
        config['seq_len'], 
        config['d_model']
    )
    return model


def train_model(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_path(config, config['preload'])
        print("Loading model from", model_filename)
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    for epoch in range(initial_epoch, config['epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device).long()
            decoder_input = batch['decoder_input'].to(device).long()
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device).long()

            # Run tensors thru model
            encoder_input = model.src_embed(encoder_input)
            encoder_output = model.encoder(encoder_input, encoder_mask)

            # decoder_input = model.tgt_embed(decoder_input)
            # decoder_output = model.decoder(encoder_output, decoder_mask, decoder_input, decoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        evaluate_model(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg),global_step, writer)

        global_step += 1

        # Save model
        model_filename = get_weights_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)



def predict(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_input = model.src_embed(source)
    encoder_output = model.encoder(encoder_input, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:,-1])
        _ , next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat((decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)), dim=1)
        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)
def evaluate_model(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer):
    model.eval()
    val_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    total_batches = len(validation_ds)

    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch['encoder_input'].to(device).long()
            decoder_input = batch['decoder_input'].to(device).long()
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device).long()

            # Forward pass
            encoder_input = model.src_embed(encoder_input)
            encoder_output = model.encoder(encoder_input, encoder_mask)

            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            # Calculate loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            val_loss += loss.item()

    # Average validation loss
    val_loss /= total_batches
    print_msg(f"Validation Loss: {val_loss:.4f}")
    writer.add_scalar('Loss/val', val_loss, global_step)
    writer.flush()

    return val_loss


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)




