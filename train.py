from scipy.signal import savgol_filter
import time
from encoder_decoder import *
from vqvae import *
from datasets import prepare_dataset
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

batch_size = 256
num_training_updates = 15000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

training_data, _, training_loader, validation_loader = prepare_dataset(batch_size)

data_variance = np.var(training_data.data / 255.0)

class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()

        self._encoder = Encoder(3, num_hiddens, num_residual_layers, num_residual_hiddens)

        self._pred_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)

        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)

        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)


    def forward(self, x):
        z = self._encoder(x)
        z = self._pred_vq_conv(z)

        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

# Training
model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

writer = SummaryWriter()
model.train()
best_recon = 1.0
train_res_recon_error = []
train_res_perplexity = []

for i in tqdm(range(num_training_updates), desc="Training..."):
    time.sleep(0.5)
    (data, _) = next(iter(training_loader))
    data = data.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss
    writer.add_scalar('Loss/Iteration', loss, i)
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

#    writer.add_scalar('Train_res_recon_error/Iteration', train_res_recon_error, i)
#    writer.add_scalar('Train_res_perplexity/Iteration', train_res_perplexity, i)

    if (i + 1) % 100 == 0:
        print(f'\nIteration: [{i+1}/{num_training_updates}]')
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print('*'*50)

        if best_recon > np.mean(train_res_recon_error[-100:]):
            best_recon = np.mean(train_res_recon_error[-100:])
            torch.save(model.state_dict(), 'model.pt')

writer.flush()
writer.close()

#Plot
train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)

f = plt.figure(figsize=(16,8))
ax = f.add_subplot(1,2,1)
ax.plot(train_res_recon_error_smooth)
ax.set_yscale('log')
ax.set_title('Smoothed NMSE.')
ax.set_xlabel('iteration')

ax = f.add_subplot(1,2,2)
ax.plot(train_res_perplexity_smooth)
ax.set_title('Smoothed Average codebook usage (perplexity).')
ax.set_xlabel('iteration')

# View reconstruction

model.eval()

(valid_originals, _) = next(iter(validation_loader))
valid_originals = valid_originals.to(device)

vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
_, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
valid_reconstructions = model._decoder(valid_quantize)

(train_originals, _) = next(iter(training_loader))
train_originals = train_originals.to(device)
_, train_reconstructions, _, _ = model._vq_vae(train_originals)

def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

show(make_grid(valid_reconstructions.cpu().data)+0.5, )
show(make_grid(valid_originals.cpu()+0.5))