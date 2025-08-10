import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
import math

parser = argparse.ArgumentParser(description='Minimal VITS2 training script')
parser.add_argument('--config', type=str, required=True, help='Path to config file')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Utility functions from VITS2
def sequence_mask(length: torch.Tensor, max_length=None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

class MelDataset(Dataset):
    def __init__(self, mel_dir):
        self.mel_files = [os.path.join(mel_dir, f) for f in os.listdir(mel_dir) if f.endswith('.mel.pt')]
    def __len__(self):
        return len(self.mel_files)
    def __getitem__(self, idx):
        mel = torch.load(self.mel_files[idx])
        
        # Normalize mel-spectrograms to prevent numerical instability
        # Convert to log scale and normalize
        mel = torch.clamp(mel, min=1e-5)  # Prevent log(0)
        mel = torch.log(mel)  # Convert to log scale
        
        # Normalize to roughly [-1, 1] range
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel = torch.clamp(mel, min=-3.0, max=3.0)  # Clamp to prevent extreme values
        
        return mel

def collate_fn(batch):
    """Custom collate function to handle variable-length mel-spectrograms"""
    # Filter out any None or corrupted mels
    valid_batch = []
    for mel in batch:
        if mel is not None and not torch.isnan(mel).any() and not torch.isinf(mel).any():
            # Additional validation for extreme values
            if mel.abs().max() < 100.0:  # Reasonable range check
                valid_batch.append(mel)
            else:
                print(f"Warning: Skipping mel with extreme values (max: {mel.abs().max():.6f})")
        else:
            print("Warning: Skipping corrupted mel spectrogram")
    
    if len(valid_batch) == 0:
        print("Error: No valid mels in batch!")
        return None, None
    
    # Find max length in batch
    max_len = max([mel.shape[-1] for mel in valid_batch])
    
    # Pad all mels to max length
    padded_mels = []
    lengths = []
    for mel in valid_batch:
        if mel.dim() == 3 and mel.shape[0] == 1:
            mel = mel.squeeze(0)  # Remove batch dim if present
        if mel.dim() == 2 and mel.shape[0] != 80:
            mel = mel.transpose(0, 1)  # Ensure (n_mels, time)
        
        # Final validation after reshaping
        if torch.isnan(mel).any() or torch.isinf(mel).any():
            print("Warning: NaN/Inf detected after reshaping, skipping")
            continue
        
        # Pad to max length
        pad_amount = max_len - mel.shape[-1]
        if pad_amount > 0:
            mel = torch.nn.functional.pad(mel, (0, pad_amount))
        
        padded_mels.append(mel)
        lengths.append(mel.shape[-1] - pad_amount)  # Original length
    
    if len(padded_mels) == 0:
        print("Error: No valid mels after processing!")
        return None, None
    
    # Stack into batch
    batch_mels = torch.stack(padded_mels, dim=0)
    batch_lengths = torch.LongTensor(lengths)
    
    return batch_mels, batch_lengths

class VITS2(torch.nn.Module):
    """
    Simplified VITS2 for mel-spectrogram training
    Based on the original VITS2 SynthesizerTrn architecture
    """
    def __init__(self, 
                 spec_channels=80, 
                 inter_channels=192, 
                 hidden_channels=256,
                 filter_channels=1024,
                 n_heads=8,
                 n_layers=6,
                 kernel_size=3,
                 p_dropout=0.1,
                 segment_size=8192,
                 gin_channels=0):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.segment_size = segment_size
        
        # Posterior Encoder (encodes mel-spectrograms to latent)
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 
            kernel_size=5, dilation_rate=1, n_layers=16,
            gin_channels=gin_channels
        )
        
        # Generator/Decoder (converts latent to audio waveform)
        self.dec = Generator(
            inter_channels,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            gin_channels=gin_channels
        )
        
        # Normalizing Flow for variational modeling
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4,
            n_flows=4, gin_channels=gin_channels
        )

class PosteriorEncoder(torch.nn.Module):
    """Proper VITS2 Posterior Encoder"""
    def __init__(self, in_channels, out_channels, hidden_channels, 
                 kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.out_channels = out_channels

        self.pre = torch.nn.Linear(in_channels, hidden_channels)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = torch.nn.Linear(hidden_channels, out_channels * 2)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        
        # Clamp input to prevent extreme values
        x = torch.clamp(x, min=-5.0, max=5.0)
        
        x = self.pre(x.transpose(1, 2)).transpose(1, 2) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x.transpose(1, 2)).transpose(1, 2) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        
        # Clamp mean and logs to prevent overflow
        m = torch.clamp(m, min=-5.0, max=5.0)
        logs = torch.clamp(logs, min=-5.0, max=2.0)
        
        # More conservative sampling
        z = m + torch.randn_like(m) * torch.exp(logs * 0.5) * x_mask
        z = torch.clamp(z, min=-10.0, max=10.0)
        
        return z, m, logs, x_mask

class WN(torch.nn.Module):
    """Proper WaveNet implementation from VITS2"""
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Linear(gin_channels, 2 * hidden_channels * n_layers)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, 
                                     dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            res_skip_layer = torch.nn.Linear(hidden_channels, res_skip_channels)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g.transpose(1, 2)).transpose(1, 2)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts.transpose(1, 2)).transpose(1, 2)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

class Generator(torch.nn.Module):
    """Simplified Generator/Vocoder"""
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, 
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel, 
                 upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        
        self.ups = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(torch.nn.ConvTranspose1d(
                upsample_initial_channel // (2**i), 
                upsample_initial_channel // (2**(i+1)), 
                k, u, padding=(k-u)//2))
        
        self.conv_post = torch.nn.Conv1d(upsample_initial_channel // (2**len(upsample_rates)), 1, 7, 1, padding=3)
        
        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
            
        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

class ResidualCouplingBlock(torch.nn.Module):
    """Proper VITS2 Normalizing Flow"""
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, 
                 n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.flows = torch.nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, 
                                                  dilation_rate, n_layers, gin_channels=gin_channels))
            self.flows.append(Flip())

    def forward(self, x, m, logs, x_mask, g=None, reverse=False):
        if reverse:
            for flow in reversed(self.flows):
                x, m, logs = flow(x, m, logs, x_mask, g=g, reverse=reverse)
        else:
            for flow in self.flows:
                x, m, logs = flow(x, m, logs, x_mask, g=g, reverse=reverse)
        return x, m, logs


class ResidualCouplingLayer(torch.nn.Module):
    """Proper VITS2 Coupling Layer"""
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, 
                 p_dropout=0, gin_channels=0, mean_only=False):
        super().__init__()
        assert channels % 2 == 0, "channels should be divisible by 2"
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = torch.nn.Linear(self.half_channels, hidden_channels)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, 
                     p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = torch.nn.Linear(hidden_channels, self.half_channels * (2 - mean_only))
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, m, logs, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        m0, m1 = torch.split(m, [self.half_channels] * 2, 1)
        logs0, logs1 = torch.split(logs, [self.half_channels] * 2, 1)
        
        h = self.pre(x0.transpose(1, 2)).transpose(1, 2) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h.transpose(1, 2)).transpose(1, 2) * x_mask
        
        if not self.mean_only:
            m_flow, logs_flow = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m_flow = stats
            logs_flow = torch.zeros_like(m_flow)

        # Clamp flows to prevent overflow
        m_flow = torch.clamp(m_flow, min=-5.0, max=5.0)
        logs_flow = torch.clamp(logs_flow, min=-5.0, max=2.0)

        if reverse:
            x1 = (x1 - m_flow) * torch.exp(-logs_flow) * x_mask
            m1 = (m1 - m_flow) * torch.exp(-logs_flow) * x_mask
            logs1 = logs1 - logs_flow
        else:
            x1 = m_flow + x1 * torch.exp(logs_flow) * x_mask
            m1 = m_flow + m1 * torch.exp(logs_flow) * x_mask
            logs1 = logs1 + logs_flow

        # Clamp outputs
        x1 = torch.clamp(x1, min=-10.0, max=10.0)
        m1 = torch.clamp(m1, min=-10.0, max=10.0)
        logs1 = torch.clamp(logs1, min=-10.0, max=10.0)

        x = torch.cat([x0, x1], 1)
        m = torch.cat([m0, m1], 1)
        logs = torch.cat([logs0, logs1], 1)
        return x, m, logs


class Flip(torch.nn.Module):
    """Flip layer for normalizing flow"""
    def forward(self, x, m, logs, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        m = torch.flip(m, [1])
        logs = torch.flip(logs, [1])
        return x, m, logs

def train():
    mel_dir = os.path.join(args.output_dir, 'mcep')
    if not os.path.exists(mel_dir):
        print(f"Mel directory {mel_dir} not found.")
        return
    
    # Initialize wandb
    wandb.init(
        project="falcon",
        name=f"vits2-{os.path.basename(args.output_dir)}",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 1e-3,
            "model": "VITS2",
            "dataset": "ljspeech",
        }
    )
    
    dataset = MelDataset(mel_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = VITS2()
    
    # Initialize model weights properly using VITS2 style initialization
    def init_weights(m, mean=0.0, std=0.01):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(mean, std)
        elif classname.find("Linear") != -1:
            m.weight.data.normal_(mean, std)
            if m.bias is not None:
                m.bias.data.fill_(0)
    
    model.apply(init_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))  # Lower learning rate
    loss_fn = torch.nn.MSELoss()
    
    step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        valid_batches = 0
        for mel_batch, mel_lengths in dataloader:
            # Skip corrupted batches
            if mel_batch is None or mel_lengths is None:
                print("Skipping corrupted batch")
                continue
                
            optimizer.zero_grad()
            
            # Forward pass through VITS2
            z_q, m_q, logs_q, y_mask = model.enc_q(mel_batch, mel_lengths)
            
            # Debug first batch
            if step == 0:
                print(f"Input mel stats: min={mel_batch.min():.6f}, max={mel_batch.max():.6f}, mean={mel_batch.mean():.6f}")
                print(f"Initial z_q stats: min={z_q.min():.6f}, max={z_q.max():.6f}, mean={z_q.mean():.6f}")
                print(f"Initial m_q stats: min={m_q.min():.6f}, max={m_q.max():.6f}, mean={m_q.mean():.6f}")
                print(f"Initial logs_q stats: min={logs_q.min():.6f}, max={logs_q.max():.6f}, mean={logs_q.mean():.6f}")
            
            z_p, m_p, logs_p = model.flow(z_q, m_q, logs_q, y_mask, reverse=True)
            
            # Debug first batch
            if step == 0:
                print(f"After flow z_p stats: min={z_p.min():.6f}, max={z_p.max():.6f}, mean={z_p.mean():.6f}")
                print(f"After flow m_p stats: min={m_p.min():.6f}, max={m_p.max():.6f}, mean={m_p.mean():.6f}")
                print(f"After flow logs_p stats: min={logs_p.min():.6f}, max={logs_p.max():.6f}, mean={logs_p.mean():.6f}")
            
            # Clamp values to prevent overflow/underflow
            logs_q = torch.clamp(logs_q, min=-10.0, max=10.0)
            logs_p = torch.clamp(logs_p, min=-10.0, max=10.0)
            m_q = torch.clamp(m_q, min=-10.0, max=10.0)
            m_p = torch.clamp(m_p, min=-10.0, max=10.0)
            z_q = torch.clamp(z_q, min=-10.0, max=10.0)
            z_p = torch.clamp(z_p, min=-10.0, max=10.0)
            
            # Simple reconstruction loss only for now
            masked_z_p = z_p * y_mask
            masked_z_q = z_q * y_mask
            
            # Compute loss on valid elements only
            num_valid = y_mask.sum()
            if num_valid > 0:
                loss = torch.sum((masked_z_p - masked_z_q) ** 2) / num_valid
            else:
                print("Warning: No valid elements in mask, skipping batch")
                continue
            
            # Add small regularization
            reg_loss = 0.01 * (torch.mean(z_q ** 2) + torch.mean(z_p ** 2))
            loss = loss + reg_loss
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss detected at step {step}! Skipping batch.")
                print(f"loss: {loss:.6f}, reg_loss: {reg_loss:.6f}")
                print(f"z_p stats: min={z_p.min():.6f}, max={z_p.max():.6f}, mean={z_p.mean():.6f}")
                print(f"z_q stats: min={z_q.min():.6f}, max={z_q.max():.6f}, mean={z_q.mean():.6f}")
                print(f"num_valid: {num_valid}")
                continue
                
            loss.backward()
            
            # Check gradients for NaN/Inf
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"NaN/Inf gradient detected in {name}")
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                print(f"Skipping step {step} due to NaN/Inf gradients")
                optimizer.zero_grad()
                continue
            
            # Gradient clipping to prevent explosion
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # Check if gradients are too large
            if grad_norm > 10.0:
                print(f"Large gradient norm detected: {grad_norm:.6f}, skipping step")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            step += 1
            
            # Console logging every 10 steps
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}")
            
            # Log to wandb every step
            wandb.log({"train_loss": loss.item(), "step": step})
            
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f} (from {valid_batches} valid batches)")
            wandb.log({"epoch": epoch+1, "avg_train_loss": avg_loss, "valid_batches": valid_batches})
        else:
            print(f"Epoch {epoch+1}/{args.epochs}: No valid batches!")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'vits2_model.pt'))
    
    # Finish wandb run
    wandb.finish()
    print(f"Training complete. Model saved to {args.output_dir}/vits2_model.pt")

if __name__ == "__main__":
    train()
