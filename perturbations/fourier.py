"""
Adapted from https://github.com/YanchaoYang/FDA/blob/master/utils/__init__.py
"""

import numpy as np
import torch
import torch
import torch.nn.functional as F


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:, :, :, :, 0]**2 + fft_im[:, :, :, :, 1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
    return fft_amp, fft_pha


def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)     # get b
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]      # top left
    amp_src[:, :, 0:b, w - b:w] = amp_trg[:, :, 0:b, w - b:w]    # top right
    amp_src[:, :, h - b:h, 0:b] = amp_trg[:, :, h - b:h, 0:b]    # bottom left
    amp_src[:, :, h - b:h, w - b:w] = amp_trg[:, :, h - b:h, w - b:w]  # bottom right
    return amp_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.rfft(src_img.clone(), signal_ndim=2, onesided=False)
    fft_trg = torch.rfft(trg_img.clone(), signal_ndim=2, onesided=False)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float)
    fft_src_[:, :, :, :, 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:, :, :, :, 1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft(fft_src_, signal_ndim=2,
                             onesided=False, signal_sizes=[imgH, imgW])

    return src_in_trg


@torch.no_grad()
def fourier_mix(src_images, tgt_images, L=0.1):
    """ Transfers style of style images to content images. Assumes input 
        is a PyTorch tensor with a batch dimension."""
    B, sC, sH, sW = src_images.shape
    B, tC, tH, tW = tgt_images.shape
    if (sH > tH) or (sW > tW):
        tgt_images = F.interpolate(tgt_images, size=(sH, sW), mode='bicubic')
    mixed_images = FDA_source_to_target(src_images, tgt_images, L=L)
    return mixed_images.to(src_images.device)
