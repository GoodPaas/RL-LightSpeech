import torch
import torch.nn as nn
from torch import optim

import os
import time
import argparse
import numpy as np

from multiprocessing import cpu_count

from networks import LightSpeech
from dataset import DataLoader, collate_fn
from dataset import LightSpeechDataset
from loss import LigthSpeechLoss

import utils
import hparams as hp


def main(args):
    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

    # Define model
    model = LightSpeech().to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of LightSpeech Parameters:', num_param)

    # Get dataset
    dataset = LightSpeechDataset()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hp.learning_rate,
                                 weight_decay=hp.weight_decay)
    actor_optimizer = torch.optim.Adam([{"params": model.embeddings.parameters()},
                                        {"params": model.pre_gru.parameters()},
                                        {"params": model.pre_linear.parameters()},
                                        {"params": model.LR.parameters()}],
                                       lr=hp.learning_rate,
                                       weight_decay=hp.weight_decay)

    # Criterion
    criterion = LigthSpeechLoss()

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Init logger
    if not os.path.exists(hp.logger_path):
        os.mkdir(hp.logger_path)

    # Define Some Information
    Time = np.array([])
    Start = time.clock()

    # Training
    model = model.train()

    for epoch in range(hp.epochs):
        # Get Training Loader
        training_loader = DataLoader(dataset,
                                     batch_size=hp.batch_size**2,
                                     shuffle=True,
                                     collate_fn=collate_fn,
                                     drop_last=True,
                                     num_workers=cpu_count())
        total_step = hp.epochs * len(training_loader) * hp.batch_size

        for i, batchs in enumerate(training_loader):
            for j, data_of_batch in enumerate(batchs):
                start_time = time.clock()

                current_step = i * hp.batch_size + j + args.restore_step + \
                    epoch * len(training_loader)*hp.batch_size + 1

                # Get Data
                character = torch.from_numpy(
                    data_of_batch["text"]).long().to(device)
                mel_gt_target = torch.from_numpy(
                    data_of_batch["mel_gt_target"]).float().to(device)
                # mel_tac2_target = torch.from_numpy(
                #     data_of_batch["mel_tac2_target"]).float().to(device)

                # D = torch.from_numpy(data_of_batch["D"]).int().to(device)
                # cemb = torch.from_numpy(
                #     data_of_batch["cemb"]).float().to(device)

                input_lengths = torch.from_numpy(
                    data_of_batch["length_text"]).long().to(device)
                output_lengths = torch.from_numpy(
                    data_of_batch["length_mel"]).long().to(device)

                max_c_len = max(input_lengths).item()
                max_mel_len = max(output_lengths).item()

                # Forward
                mel, P, predicted_length, history = model(character,
                                                          input_lengths,
                                                          max_c_len,
                                                          max_mel_len)

                # print(predicted_length)

                # Cal Loss
                mel_loss, len_loss, pg_loss = criterion(mel,
                                                        predicted_length,
                                                        mel_gt_target,
                                                        output_lengths,
                                                        P, history)
                # model_loss = mel_loss + len_loss
                model_loss = mel_loss
                actor_loss = pg_loss

                # print(mel_loss, len_loss, pg_loss)

                # Logger
                m_l = mel_loss.item()
                l_l = len_loss.item()
                p_l = pg_loss.item()

                with open(os.path.join("logger", "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l)+"\n")

                with open(os.path.join("logger", "len_loss.txt"), "a") as f_l_loss:
                    f_l_loss.write(str(l_l)+"\n")

                with open(os.path.join("logger", "pg_loss.txt"), "a") as f_p_loss:
                    f_p_loss.write(str(p_l)+"\n")

                # Backward
                model_loss.backward(retain_graph=True)
                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                # Update weights
                optimizer.step()
                # Init
                optimizer.zero_grad()

                actor_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                actor_optimizer.step()
                actor_optimizer.zero_grad()

                # Print
                if current_step % hp.log_step == 0:
                    Now = time.clock()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    str2 = "Mel Loss: {:.4f}, Length Loss: {:.4f}, Policy Loss: {:.4f};".format(
                        m_l, l_l, p_l)
                    str3 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)

                    with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                        f_logger.write(str1 + "\n")
                        f_logger.write(str2 + "\n")
                        f_logger.write(str3 + "\n")
                        f_logger.write("\n")

                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)

                end_time = time.clock()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()

    main(args)
