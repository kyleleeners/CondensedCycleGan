from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchaudio

import models
import torch
from torch import nn
from torch.utils import data
import torchaudio.transforms as transforms
import utils
from dataloader import AudioFolder
from stft import istft


""" gpu """
gpu_id = [0]
utils.cuda_devices(gpu_id)
torch.cuda.empty_cache()


""" param """
epochs = 300
batch_size = 1
lr = 0.0002
dataset_dir = 'datasets/music2music'


""" data """
length = 101024

transform=transforms.Compose([
    transforms.PadTrim(length,0),
    transforms.DownmixMono(True)
])

dataset_dirs = utils.reorganize(dataset_dir)
a_data = AudioFolder(dataset_dirs['trainA'], transform=transform)
b_data = AudioFolder(dataset_dirs['trainB'], transform=transform)
a_test_data = AudioFolder(dataset_dirs['testA'], transform=transform)
b_test_data = AudioFolder(dataset_dirs['testB'], transform=transform)
a_loader = data.DataLoader(a_data, batch_size=batch_size, shuffle=True)
b_loader = data.DataLoader(b_data, batch_size=batch_size, shuffle=True)
a_test_loader = data.DataLoader(a_test_data, batch_size=1, shuffle=True)
b_test_loader = data.DataLoader(b_test_data, batch_size=1, shuffle=True)

a_fake_pool = utils.ItemPool()
b_fake_pool = utils.ItemPool()

""" model """
Da = models.Discriminator()
Db = models.Discriminator()
Ga = models.Generator()
Gb = models.Generator()
MSE = nn.MSELoss()
L1 = nn.L1Loss()
utils.cuda([Da, Db, Ga, Gb])

da_optimizer = torch.optim.Adam(Da.parameters(), lr=lr, betas=(0.5, 0.999))
db_optimizer = torch.optim.Adam(Db.parameters(), lr=lr, betas=(0.5, 0.999))
ga_optimizer = torch.optim.Adam(Ga.parameters(), lr=lr, betas=(0.5, 0.999))
gb_optimizer = torch.optim.Adam(Gb.parameters(), lr=lr, betas=(0.5, 0.999))

""" load checkpoint """
ckpt_dir = './checkpoints/music2music'
utils.mkdir([ckpt_dir])
try:
    ckpt = utils.load_checkpoint(ckpt_dir)
    start_epoch = ckpt['epoch']
    Da.load_state_dict(ckpt['Da'])
    Db.load_state_dict(ckpt['Db'])
    Ga.load_state_dict(ckpt['Ga'])
    Gb.load_state_dict(ckpt['Gb'])
    da_optimizer.load_state_dict(ckpt['da_optimizer'])
    db_optimizer.load_state_dict(ckpt['db_optimizer'])
    ga_optimizer.load_state_dict(ckpt['ga_optimizer'])
    gb_optimizer.load_state_dict(ckpt['gb_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0

""" run """
a_real_test = iter(a_test_loader).next()[0]
b_real_test = iter(b_test_loader).next()[0]
a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
n_fft = 2050

# helpers

def convert_to_spectro(vec):
    squeeze = vec.squeeze(1)
    stft = torch.stft(squeeze, n_fft)
    return torch.transpose(stft, dim0=1, dim1=3)

def convert_from_spectro(vec):
    trans = torch.transpose(vec, dim0=1, dim1=3)
    back = istft(trans, n_fft//4, length=length)
    return back.unsqueeze(1)

for epoch in range(start_epoch, epochs):
    for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
        # step
        step = epoch * min(len(a_loader), len(b_loader)) + i + 1

        # leaves
        a_real = a_real[0]
        b_real = b_real[0]

        #convert to spectro
        a_stft = convert_to_spectro(a_real)
        b_stft = convert_to_spectro(b_real)

        a_stft, b_stft = utils.cuda([a_stft, b_stft])

        # set train
        Ga.train()
        Gb.train()

        # train G
        a_fake = Ga(b_stft)
        b_fake = Gb(a_stft)

        a_rec = Ga(b_fake)
        b_rec = Gb(a_fake)

        # gen losses
        a_f_dis = Da(a_fake)
        b_f_dis = Db(b_fake)
        r_label = utils.cuda(torch.ones(a_f_dis.size()))
        a_gen_loss = MSE(a_f_dis, r_label)
        b_gen_loss = MSE(b_f_dis, r_label)

        # rec losses
        a_rec_loss = L1(a_rec, a_stft)
        b_rec_loss = L1(b_rec, b_stft)

        # g loss
        g_loss = a_gen_loss + b_gen_loss + a_rec_loss * 10.0 + b_rec_loss * 10.0

        # backward
        Ga.zero_grad()
        Gb.zero_grad()
        g_loss.backward()
        ga_optimizer.step()
        gb_optimizer.step()

        # leaves
        a_fake = torch.Tensor(a_fake_pool([a_fake.cpu().data.numpy()])[0])
        b_fake = torch.Tensor(b_fake_pool([b_fake.cpu().data.numpy()])[0])
        a_fake, b_fake = utils.cuda([a_fake, b_fake])

        # train D
        a_r_dis = Da(a_stft)
        a_f_dis = Da(a_fake)
        b_r_dis = Db(b_stft)
        b_f_dis = Db(b_fake)
        r_label = utils.cuda(torch.ones(a_f_dis.size()))
        f_label = utils.cuda(torch.zeros(a_f_dis.size()))

        # d loss
        a_d_r_loss = MSE(a_r_dis, r_label)
        a_d_f_loss = MSE(a_f_dis, f_label)
        b_d_r_loss = MSE(b_r_dis, r_label)
        b_d_f_loss = MSE(b_f_dis, f_label)

        a_d_loss = (a_d_r_loss + a_d_f_loss) * 0.5
        b_d_loss = (b_d_r_loss + b_d_f_loss) * 0.5

        # backward
        Da.zero_grad()
        Db.zero_grad()
        a_d_loss.backward()
        b_d_loss.backward()
        da_optimizer.step()
        db_optimizer.step()

        torch.cuda.empty_cache()

        if (i + 1) % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, i + 1, min(len(a_loader), len(b_loader))))
            print("A discrim loss: %5f Gen loss: %5f rec loss: %5f" % (a_d_loss, a_gen_loss, a_rec_loss))
            print("B discrim loss: %5f Gen loss: %5f rec loss: %5f" % (b_d_loss, b_gen_loss, b_rec_loss))

        if (i + 1) % min(len(a_loader), len(b_loader)) == 0:

            nets = [Ga, Gb]
            for net in nets:
                for param in net.parameters():
                    param.requires_grad = False

            #Try a new song
            a_real_test = iter(a_test_loader).next()[0]
            b_real_test = iter(b_test_loader).next()[0]
            a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])

            #convert to spectro
            a_real_test_spectro = convert_to_spectro(a_real_test)
            b_real_test_spectro = convert_to_spectro(b_real_test)

            # train G
            a_fake_test = Ga(b_real_test_spectro)
            b_fake_test = Gb(a_real_test_spectro)

            a_rec_test = Ga(b_fake_test)
            b_rec_test = Gb(a_fake_test)

            #convert back
            a_fake_test = convert_from_spectro(a_fake_test)
            b_fake_test = convert_from_spectro(b_fake_test)
            a_rec_test = convert_from_spectro(a_rec_test)
            b_rec_test = convert_from_spectro(b_rec_test)

            for net in nets:
                for param in net.parameters():
                    param.requires_grad = True

            a_real_test_cpu = a_real_test.squeeze(0).cpu()
            a_fake_test_cpu = a_fake_test.squeeze(0).cpu()
            a_rec_test_cpu = a_rec_test.squeeze(0).cpu()

            b_real_test_cpu = b_real_test.squeeze(0).cpu()
            b_fake_test_cpu = b_fake_test.squeeze(0).cpu()
            b_rec_test_cpu = b_rec_test.squeeze(0).cpu()

            save_dir = './samples_while_training/epoch_%d' % epoch
            utils.mkdir([save_dir])

            torchaudio.save('%s/a_real.wav' % save_dir, a_real_test_cpu, 44100)
            torchaudio.save('%s/a_fake.wav' % save_dir, a_fake_test_cpu, 44100)
            torchaudio.save('%s/a_rec.wav' % save_dir, a_rec_test_cpu, 44100)
            torchaudio.save('%s/b_real.wav' % save_dir, b_real_test_cpu, 44100)
            torchaudio.save('%s/b_fake.wav' % save_dir, b_fake_test_cpu, 44100)
            torchaudio.save('%s/b_rec.wav' % save_dir, b_rec_test_cpu, 44100)

    utils.save_checkpoint({'epoch': epoch + 1,
                           'Da': Da.state_dict(),
                           'Db': Db.state_dict(),
                           'Ga': Ga.state_dict(),
                           'Gb': Gb.state_dict(),
                           'da_optimizer': da_optimizer.state_dict(),
                           'db_optimizer': db_optimizer.state_dict(),
                           'ga_optimizer': ga_optimizer.state_dict(),
                           'gb_optimizer': gb_optimizer.state_dict()},
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                          max_keep=2)