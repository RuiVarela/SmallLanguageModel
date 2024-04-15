import argparse
import csv
import json
import logging
import math
import os
import time

import torch

from model import GPTConfig, GPT
from tokenizer import Tokenizer
from dataset import TokenDataset, generate_dataloader

import matplotlib
import matplotlib.pyplot as plt

#
# Training code
#

def generate_stats(iter, losses, checkpoint_file, create_stats_files):
    csv_file = os.path.join(os.path.dirname(checkpoint_file), "run.csv")
    png_file = os.path.join(os.path.dirname(checkpoint_file), "run.png")

    csv_fieldnames = ['iter', 'train loss', 'test loss']
    train_loss = losses['train'].item()
    test_loss = losses['val'].item()

    if create_stats_files and os.path.exists(csv_file):
        os.remove(csv_file)

    #
    # Save csv
    #
    add_header = not os.path.exists(csv_file)
    with open(csv_file,'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        if add_header:
            writer.writeheader()
        writer.writerow({'iter': iter, 'train loss': train_loss, 'test loss': test_loss})


    #
    # Generate plot
    #
    counter = 0
    data_count = []
    data_iter = []
    data_train_loss = []
    data_test_loss = []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f, fieldnames=csv_fieldnames)
        next(reader, None)  # skip the headers
        for row in reader:
            data_count.append(counter)
            counter = counter + 1
            data_iter.append(int(row['iter']))
            data_train_loss.append(float(row['train loss']))
            data_test_loss.append(float(row['test loss']))

    matplotlib.use('Agg')

    plt.clf()
    plt.plot(data_count, data_train_loss, color='red', label='Train Loss')
    plt.plot(data_count, data_test_loss, color='blue', label='Test Loss')
    plt.xticks(data_count, ['{}'.format(i+1) for i in data_count])
    plt.xlabel('X')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.4)
    plt.savefig(png_file)


def estimate_loss(eval_iters, model, train_loader, test_loader, device):
    out = {}
    model.eval()
    for split, loader in zip(['train', 'val'], [iter(train_loader), iter(test_loader)]):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(loader)
            X = X.to(device)
            Y = Y.to(device)

            with torch.no_grad():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, settings):
    warmup_iters = settings["warmup_iters"]
    learning_rate = settings["learning_rate"]
    lr_decay_iters = settings["lr_decay_iters"]
    min_lr = settings["min_lr"]

    if not settings["decay_lr"]:
        return learning_rate

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def train(checkpoint_file, data_tokenized, settings, resume, load):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logging.info(f"tokenizer {data_tokenized}")

    gpt_config = GPTConfig()
    gpt_config.block_size = settings["block_size"]
    gpt_config.vocab_size = settings["vocabulary_size"]
    gpt_config.n_layer = settings["n_layer"]
    gpt_config.n_head = settings["n_head"]
    gpt_config.n_embd = settings["n_embd"]
    gpt_config.dropout = settings["dropout"]
    logging.info(f"gpt_config {gpt_config}")

    model = GPT(gpt_config)

    iter_num = 0
    best_val_loss = 1e9
    create_stats_files = True

    if resume or load:
        logging.info(f"Loading checkpoint {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)

        if resume:
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
            create_stats_files = False

    model = model.to(device)

    # optimizer
    weight_decay = settings["weight_decay"]
    learning_rate = settings["learning_rate"]
    beta1 = settings["beta1"]
    beta2 = settings["beta2"]
    decay_lr = settings["decay_lr"]
    logging.info(f"optimizer weight_decay={weight_decay} learning_rate={learning_rate} beta1={beta1} beta2={beta2} decay_lr={decay_lr}")

    train_loader = generate_dataloader(data_tokenized, "train", settings)
    test_loader = generate_dataloader(data_tokenized, "train", settings)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2))
    if resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    model.train()

    iter_time = time.time()
    data_iter = iter(train_loader)

    logging.info(f"Starting train iter_num={iter_num} best_val_loss={best_val_loss}")

    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, settings)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # fetch the next batch (x, y) and re-init iterator if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            logging.info("stop iteration start")
            data_iter = iter(train_loader)
            batch = next(data_iter)
            logging.info("stop iteration end")

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        # forward the model
        logits, loss = model(x, y)

        # backprop and update the parameters
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_norm_clip"])
        optimizer.step()

        #self.trigger_callbacks('on_batch_end')
        iter_num += 1
        tnow = time.time()
        iter_dt = tnow - iter_time
        iter_time = tnow

        if iter_num % settings["eval_interval"] == 0: 
            losses = estimate_loss(settings["eval_iters"], model, train_loader, test_loader, device)
            logging.info(f"step {iter_num}: lr {lr:7.5} train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            generate_stats(iter_num, losses, checkpoint_file, create_stats_files)
            create_stats_files = False

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'model_args': gpt_config,
                    }
                    logging.info(f"saving checkpoint to {checkpoint_file}")
                    torch.save(checkpoint, checkpoint_file)
        elif iter_num % 100 == 0:
            logging.info(f"step {iter_num:06}: batch_time {iter_dt:7.5} lr {lr:7.5} loss: {loss.item():.4f}")

        # termination conditions"
        if iter_num >= settings["max_iters"]:
            break   

#
# inference code
#
def complete(tokenizer, checkpoint_file, prompt, max_tokens, temperature, top_k):
    logging.info(f"loading model {checkpoint_file}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_file)
    gptconf = checkpoint['model_args']
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    logging.info(f"max_tokens={max_tokens} temperature={temperature} top_k={top_k} prompt={prompt}")
    logging.info(f"-------")
    # encode the beginning of the prompt
    start_ids = tokenizer.encode(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        y = model.generate(x, max_tokens, temperature=temperature, top_k=top_k)
        y_tokens = y[0].tolist()
        text = tokenizer.decode(y_tokens)
        logging.info(text)


#
# export 
#
def export(checkpoint_file, settings):
    # filename without extension
    target = os.path.splitext(checkpoint_file)[0] + ".onnx"
    logging.info(f"Exporting model to ONNX format. {checkpoint_file} -> {target}")


    checkpoint = torch.load(checkpoint_file)
    gptconf = checkpoint['model_args']
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

    model.eval()

    input_size = (1, settings["block_size"])

    x = torch.randint(0, settings['vocabulary_size'], input_size)
    logging.info(f"{x.shape}")


    # Export the model
    torch.onnx.export(model,             # model being run
                x,                         # model input (or a tuple for multiple inputs)
                target,                    # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=17,          # the ONNX version to export the model to
                do_constant_folding=False,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output'], # the model's output names
                dynamic_axes={'input' : {1 : 'tokens'},    # variable length axes
                              #'output' : {1 : 'tokens'}
                              }
                )

def configure_log(project_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.handlers.clear()
    
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(project_dir, 'log.txt'))
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='sml', description='Small Language Model')
    parser.add_argument('--project', help="project folder", default="tinyshakespeare")

    parser.add_argument('--train_tokenizer', action='store_true', help="train the tokenizer on the dataset")
    parser.add_argument('--tokenize_message', help="tokenize a provided message")
    parser.add_argument('--tokenize_data', action='store_true', help="tokenizes the entire data for the projects and stores it binary")

    parser.add_argument('--sample_dataset', help="samples a dataset")

    parser.add_argument('--train', action='store_true', help="trains the model")
    parser.add_argument('--resume', action='store_true', help="continues training the model using the available checkpoint")
    parser.add_argument('--train_with_weights', action='store_true', help="trains the model, but initializes its weights from a checkpoint")
    parser.add_argument('--export', action='store_true', help="export to onnx")

    parser.add_argument('--temperature', type=float, help="1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions", default=0.8)
    parser.add_argument('--top_k', type=int, help="retain only the top_k most likely tokens, clamp others to have 0 probability", default=200)
    parser.add_argument('--max_tokens', type=int, help="maximum tokens to produce", default=500)

    parser.add_argument('--complete', help="completes some text")

    args = parser.parse_args()

    project = args.project
    settings_file = os.path.join(project, "project.json")
    data_file = os.path.join(project, "data.txt")
    data_tokenized = os.path.join(project, "data.bin")
    vocabulary_file = os.path.join(project, "vocabulary.json")
    checkpoint_file = os.path.join(project, "ckpt.pt")

    if not os.path.exists(settings_file):
        logging.error(f"File {settings_file} does not exist")
        exit(1)

    with open(settings_file, 'r') as f:
        settings = json.load(f)

    configure_log(project)

    tokenizer = Tokenizer(vocabulary_file)

    if args.train_tokenizer:

        tokenizer.train(data_file, vocabulary_file, settings["vocabulary_size"])

    elif args.tokenize_message:

        text = args.tokenize_message
        encoded = tokenizer.encode(text)
        logging.info(f"len:{len(text):05} text: {text} ")
        logging.info(f"len:{len(encoded):05} encoded: {encoded} ")
        decoded = tokenizer.decode(encoded)
        logging.info(f"len:{len(decoded):05} ok:{decoded==text} {tokenizer.debug(encoded)} ")

    elif args.tokenize_data:

        encoded = tokenizer.tokenize_text_file(data_file, data_tokenized)

    elif args.sample_dataset:

        loader = generate_dataloader(data_tokenized,  args.sample_dataset, settings)
        bx, by = next(iter(loader))

        logging.info(f"batch {bx.shape}, {by.shape}")
        x, y = bx[0], by[0]
        x, y = x.tolist(), y.tolist()
        logging.info(f"len:{len(x):05} x: {x} ")
        logging.info(f"-------------")
        logging.info(f"len:{len(y):05} y: {y} ")
        xd = tokenizer.decode(x)
        yd = tokenizer.decode(y)
        logging.info(f"-------------")
        logging.info(f"len:{len(xd):05} {xd}")
        logging.info(f"-------------")
        logging.info(f"len:{len(yd):05} {yd}")

    elif args.train:

        train(checkpoint_file, data_tokenized, settings, False, False)

    elif args.resume:

        train(checkpoint_file, data_tokenized, settings, True, True)

    elif args.train_with_weights:

        train(checkpoint_file, data_tokenized, settings, False, True)

    elif args.complete:

        complete(tokenizer, checkpoint_file, args.complete, args.max_tokens, args.temperature, args.top_k)

    elif args.export:

        export(checkpoint_file, settings)