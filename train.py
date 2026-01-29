import argparse
import json
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import waveglow.logging as logger

from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from scipy.io import wavfile
from waveglow.modules import UpsampleNet
from waveglow.model import WaveGlow
from waveglow.dataset import WaveGlowDataset, WaveGlowCollate
from torch.utils.tensorboard import SummaryWriter


# Basic model parameters as external flags.
FLAGS = None


def log_audio_to_tensorboard(writer, waveform, sample_rate, tag, global_step):
    """Log generated audio to TensorBoard."""
    # Normalize waveform to [-1, 1]
    waveform = torch.clamp(waveform, min=-1., max=1.)
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)  # Remove batch dimension if present
    writer.add_audio(tag, waveform, global_step, sample_rate=sample_rate)


def save_checkpoint(upsample_net, model, optimizer,
                    step, checkpoint_dir):
    checkpoint_path = os.path.join(
        checkpoint_dir, "model.ckpt-{}.pt".format(step))
    torch.save({"upsample_net": upsample_net.state_dict(),
                "waveglow": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": step}, checkpoint_path)
    logger.info("Saved checkpoint: {}".format(checkpoint_path))

    with open(os.path.join(checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write("model.ckpt-{}".format(step))


def attempt_to_restore(upsample_net, model, optimizer, checkpoint_dir):
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(
            checkpoint_dir, "{}.pt".format(checkpoint_filename))
        logger.info("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, FLAGS.use_cuda)
        upsample_net.load_state_dict(checkpoint["upsample_net"])
        model.load_state_dict(checkpoint["waveglow"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]

    else:
        global_step = 0

    return global_step


def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def validate(model, upsample_net, val_loader, device, normal, writer, global_step, params):
    """Perform validation and log results."""
    model.eval()
    upsample_net.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for sample, local_condition in val_loader:
            sample, local_condition = sample.to(device), local_condition.to(device)
            
            local_condition = upsample_net(local_condition)
            logdet = torch.zeros_like(sample[:, 0, 0])
            output, logdet = model(sample, logdet=logdet, reverse=False,
                                   local_condition=local_condition)
            
            likelihood = torch.sum(normal.log_prob(output), (1, 2))
            loss = -(likelihood + logdet).mean()
            total_loss += loss.item()
            num_batches += 1
            
            # Generate audio from first batch for visualization
            if num_batches == 1:
                noise = torch.randn_like(output)
                with torch.no_grad():
                    generated_audio, _ = model(noise, reverse=True, logdet=None,
                                          local_condition=local_condition)
                    # generated_audio shape: (batch, channels, time)
                    # Take first sample and first channel
                    audio_sample = generated_audio[0, 0, :].cpu()
                    log_audio_to_tensorboard(writer, audio_sample.unsqueeze(0), 
                                           params['waveglow']['sample_rate'],
                                           'validation/generated_audio', 
                                           global_step)
    
    val_loss = total_loss / max(num_batches, 1)
    writer.add_scalar('loss/validation', val_loss, global_step)
    logger.info('Validation loss: %.3f' % val_loss)
    
    model.train()
    upsample_net.train()
    
    return val_loss


def build_model(params):
    upsample_net = UpsampleNet(
        upsample_factor=params['upsample_net']['upsample_factor'],
        upsample_method=params['upsample_net']['upsample_method'],
        squeeze_factor=params['waveglow']['squeeze_factor'])

    input_channels = params['upsample_net']['input_channels']
    local_condition_channels = input_channels * params['waveglow']['squeeze_factor']
    model = WaveGlow(
        squeeze_factor=params['waveglow']['squeeze_factor'],
        num_layers=params['waveglow']['num_layers'],
        wn_filter_width=params['waveglow']['wn_filter_width'],
        wn_dilation_layers=params['waveglow']['wn_dilation_layers'],
        wn_residual_channels=params['waveglow']['wn_residual_channels'],
        wn_dilation_channels=params['waveglow']['wn_dilation_channels'],
        wn_skip_channels=params['waveglow']['wn_skip_channels'],
        local_condition_channels=local_condition_channels)

    return upsample_net, model


def main(_):
    device = torch.device("cuda" if FLAGS.use_cuda else "cpu")

    with open(FLAGS.waveglow_params, 'r') as f:
        params = json.load(f)

    # Determine metadata files (from config or command line)
    train_metadata_file = FLAGS.metadata_file or params.get('data', {}).get('train_metadata_file')
    val_metadata_file = params.get('data', {}).get('val_metadata_file')

    upsample_net, model = build_model(params)
    print(upsample_net)
    print(model)

    # Load train and validation datasets
    if train_metadata_file and val_metadata_file:
        # Load from separate metadata files
        train_dataset = WaveGlowDataset(audio_dir=FLAGS.audio_dir,
                                       sample_rate=params['waveglow']['sample_rate'],
                                       local_condition_enabled=True,
                                       local_condition_dir=FLAGS.local_condition_dir,
                                       metadata_file=train_metadata_file)
        val_dataset = WaveGlowDataset(audio_dir=FLAGS.audio_dir,
                                     sample_rate=params['waveglow']['sample_rate'],
                                     local_condition_enabled=True,
                                     local_condition_dir=FLAGS.local_condition_dir,
                                     metadata_file=val_metadata_file)
        logger.info('Loaded train dataset from {}: {} samples'.format(train_metadata_file, len(train_dataset)))
        logger.info('Loaded val dataset from {}: {} samples'.format(val_metadata_file, len(val_dataset)))
        use_metadata = True
    else:
        # Load from directories and split
        dataset = WaveGlowDataset(audio_dir=FLAGS.audio_dir,
                                  sample_rate=params['waveglow']['sample_rate'],
                                  local_condition_enabled=True,
                                  local_condition_dir=FLAGS.local_condition_dir)
        
        # Split dataset into train and validation (90% train, 10% validation)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        logger.info('Loaded dataset from directories: {} samples'.format(len(dataset)))
        use_metadata = False
    
    logger.info('Dataset split: {} train samples, {} validation samples'.format(len(train_dataset), len(val_dataset)))
    
    collate_fn = WaveGlowCollate(sample_size=FLAGS.sample_size,
                                 upsample_factor=params['upsample_net']['upsample_factor'],
                                 local_condition_enabled=True,
                                 use_metadata=use_metadata)
    trainloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                             shuffle=True, num_workers=FLAGS.num_workers,
                             collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size,
                           shuffle=False, num_workers=FLAGS.num_workers,
                           collate_fn=collate_fn, pin_memory=True)

    if FLAGS.use_cuda:
        logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))

    model.to(device)
    upsample_net.to(device)

    normal = Normal(loc=torch.tensor([0.0], device=device),
                    scale=torch.tensor([np.sqrt(0.5)], device=device))

    # params = list(upsample_net.parameters()) + list(model.parameters())
    # optimizer = optim.Adam(params, lr=FLAGS.learning_rate)
    model_params = list(upsample_net.parameters()) + list(model.parameters())
    optimizer = optim.Adam(model_params, lr=FLAGS.learning_rate)

    if FLAGS.restore_from is not None:
        restore_step = attempt_to_restore(upsample_net, model, optimizer,
                                          FLAGS.restore_from)

    global_step = attempt_to_restore(upsample_net, model, optimizer,
                                     FLAGS.save_dir)

    if FLAGS.restore_from is not None and global_step == 0:
        global_step = restore_step

    scheduler = StepLR(optimizer, step_size=FLAGS.decay_steps,
                       gamma=FLAGS.decay_rate, last_epoch=global_step - 1)

    writer = SummaryWriter(FLAGS.save_dir)

    for epoch in range(FLAGS.max_epochs):
        epoch_loss = 0.0
        model.train()
        upsample_net.train()
        
        for i, data in enumerate(trainloader, 0):
            sample, local_condition = data
            sample, local_condition = sample.to(device), local_condition.to(device)

            optimizer.zero_grad()

            local_condition = upsample_net(local_condition)
            logdet = torch.zeros_like(sample[:, 0, 0])
            output, logdet = model(sample, logdet=logdet, reverse=False,
                                   local_condition=local_condition)

            likelihood = torch.sum(normal.log_prob(output), (1, 2))
            loss = -(likelihood + logdet).mean()

            if (i + 1) % FLAGS.log_interval == 0:
                logger.info('[%d, %3d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

            if global_step % FLAGS.summary_interval == 0:
                writer.add_scalar('loss/training', loss.item(), global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                
                # Log gradient norms
                total_norm = 0
                for p in list(model.parameters()) + list(upsample_net.parameters()):
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar('gradient_norm', total_norm, global_step)

            epoch_loss += loss.item()

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            clip_grad_norm_(list(model.parameters()) + list(upsample_net.parameters()), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % FLAGS.checkpoint_interval == 0:
                save_checkpoint(upsample_net, model, optimizer, global_step, FLAGS.save_dir)
                logger.info('Checkpoint saved at step {}'.format(global_step))

        epoch_loss /= (i + 1)
        logger.info('[epoch %d] training loss: %.3f' % (epoch + 1, epoch_loss))
        writer.add_scalar('loss/epoch_training', epoch_loss, epoch)
        
        # Validation every N epochs
        if (epoch + 1) % FLAGS.val_interval == 0:
            logger.info('Running validation...')
            validate(model, upsample_net, val_loader, device, normal, writer, global_step, params)
            writer.flush()


if __name__ == '__main__':
    logger.set_verbosity(logger.INFO)

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Mini-batch size.')
    parser.add_argument(
        '--sample_size',
        type=int,
        default=16384,
        help='Sample size of audio clip. Must be divisible by upsample_factor.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Initial learning rate.')
    parser.add_argument(
        '--decay_steps',
        type=int,
        default=100000,
        help='Steps to decay learning rate.')
    parser.add_argument(
        '--decay_rate',
        type=float,
        default=0.95,
        help='Decay rate of learning rate.')
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=500,
        help='Max number of epochs to run trainer totally.',)
    parser.add_argument(
        '--waveglow_params',
        type=str,
        default='waveglow_params.json',
        help='JSON file with the network parameters.')
    parser.add_argument(
        '--use_cuda',
        type=_str_to_bool,
        default=True,
        help='Enables CUDA training.')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of dataloader workers.')
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=1000,
        help='Steps between writing checkpoints.')
    parser.add_argument(
        '--restore_from',
        type=str,
        default=None,
        help='Directory to restore.')
    parser.add_argument(
        '--modelDir',
        dest='save_dir',
        type=str,
        default='checkpoints/',
        help='Directory to put the training result.')
    parser.add_argument(
        '--audio_dir',
        type=str,
        default='corpus/audio',
        help='Directory of audio data.')
    parser.add_argument(
        '--local_condition_dir',
        type=str,
        default='corpus/mels',
        help='Directory of local condition data.')
    parser.add_argument(
        '--metadata_file',
        type=str,
        default=None,
        help='Training metadata file with format: path/to/mel.npy|text. One entry per line. '
             'If not provided, will use paths from config file or directories.')
    parser.add_argument(
        '--summary_interval',
        type=int,
        default=100,
        help='Steps between running summary ops.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='Steps between logging training details.')
    parser.add_argument(
        '--val_interval',
        type=int,
        default=1,
        help='Epochs between running validation.')
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
    os.makedirs(FLAGS.save_dir)
    main([sys.argv[0]] + unparsed)
