#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import PyTorch root package import torch
import torch

import math
import numpy as np


class CompressorType:
    IDENTICAL = 1                # Identical compressor
    LAZY_COMPRESSOR = 2          # Lazy or Bernulli compressor
    RANDK_COMPRESSOR = 3         # Rank-K compressor
    NATURAL_COMPRESSOR_FP32 = 4  # Natural compressor with FP32
    STANDARD_DITHERING_FP32 = 5  # Standard dithering with FP32
    NATURAL_DITHERING_FP32 = 6   # Natural Dithering applied for FP32 components vectors
    TOPK_COMPRESSOR = 7          # Top-K compressor
    RANK_K_COMPRESSOR = 8        # Rank-K compressor


class Compressor:
    """ Collection of unbiased compressor E[C(x)]=x and E[|C(x)-x|^2]<=w|x|^2 \\iff E[|C(x)|^2] <= (w+1)|x|^2 """

    def resetStats(self):
        """Reset internal statistics for compressor"""
        self.total_input_components = 0            # Total scalar component(fp32 scalar) processed by the compressor
        self.really_need_to_send_components = 0    # Total scalar component which need to be send across the network
        self.last_input_advance = 0                # Last input components through the last call
        self.last_need_to_send_advance = 0         # Last need to send scalar component.

    def __init__(self):
        """Ctor of the compressor by default compressor type is identical"""
        self.compressorType = CompressorType.IDENTICAL
        self.total_input_components = 0
        self.really_need_to_send_components = 0
        self.last_input_advance = 0
        self.last_need_to_send_advance = 0
        # remove self.w init, because self.w is undefined for biased compressors
        # self.w = 0.0

    def fullName(self):
        """Get fullname of the compressor."""
        omega = r'$\omega$'
        if self.compressorType == CompressorType.IDENTICAL:
            return f"Identical"
        if self.compressorType == CompressorType.LAZY_COMPRESSOR:
            return f"Bernoulli(Lazy) [p={self.P:g},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.RANDK_COMPRESSOR:
            return f"Rand [K={self.K},D={self.D}]"
        if self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP32:
            return f"Natural for fp32 [{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.STANDARD_DITHERING_FP32:
            return f"Standard Dithering for fp32[s={self.s}]"
        if self.compressorType == CompressorType.NATURAL_DITHERING_FP32:
            return f"Natural Dithering for fp32[s={self.s},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.TOPK_COMPRESSOR:
            return f"Top [K={self.K},D={self.D}]"
        if self.compressorType == CompressorType.RANK_K_COMPRESSOR:
            return f"Rank [K={self.K},D={self.D}]"

        return "?"

    def makeIdenticalCompressor(self):
        """ Make identical compressor """
        self.compressorType = CompressorType.IDENTICAL
        self.w = 0.0
        self.resetStats()

    def makeLazyCompressor(self, P):
        """ Make lazy compressor which with probability 'p' returns x/p and w.p. '1-p' return 0 """

        # w + 1 = p* 1/(p**2) => w = 1/p - 1
        self.compressorType = CompressorType.LAZY_COMPRESSOR
        self.P = P
        self.w = 1.0 / P - 1.0
        self.resetStats()

    def makeStandardDitheringFP32(self, D, levels, p=float("inf")):
        """
        Make standard dithering schema with having uniform levels intervals inside [0.0, 1.0]
        and using "p" norm for normalizing vectors
        """
        self.D = D
        self.compressorType = CompressorType.STANDARD_DITHERING_FP32
        # levels + 1 values in range [0.0, 1.0] which uniformly split this segment
        self.levelsValues = torch.arange(0.0, 1.0 + 1.0 / levels * 0.5, 1.0 / levels)
        self.s = len(self.levelsValues) - 1  # should be equal to level
        assert self.s == levels

        self.p = p
        self.w = 0.0  # TODO - specify W for dithering

        self.resetStats()

    def makeQSGD_FP32(self, D, levels):
        """ Make QSGD compressors with specific number of levels"""

        self.makeStandardDitheringFP32(D, levels, p=2)
        # Lemma 3.1. from https://arxiv.org/pdf/1610.02132.pdf, page 5
        self.w = min(D / (levels * levels), D ** 0.5 / levels)

    def makeTernGrad(self, D):
        """ Make Ternary Gradient compressor """
        # https://arxiv.org/pdf/1705.07878.pdf
        self.makeStandardDitheringFP32(D, levels=1, p=float("inf"))
        self.w = 0.0

    def makeNaturalDitheringFP32(self, D, levels, p=float("inf")):
        """
        Make dithering schema with having levels intervals inside [0.0, 1.0] with lengths power of (1/2)
        and using "p" norm for normalizing vectors
        """
        self.D = D
        self.compressorType = CompressorType.NATURAL_DITHERING_FP32
        self.levelsValues = torch.zeros(levels + 1)
        for i in range(levels):
            self.levelsValues[i] = (1.0 / 2.0) ** i
        self.levelsValues = torch.flip(self.levelsValues, dims=[0])
        self.s = len(self.levelsValues) - 1
        assert self.s == levels

        self.p = p

        r = min(p, 2)
        self.w = 1.0 / 8.0 + (D ** (1.0 / r)) / (2 ** (self.s - 1)) * min(1, (D ** (1.0 / r)) / (2 ** (self.s - 1)))
        self.resetStats()

    def makeRandKCompressor(self, D, K):
        """Make Random/Nice sparsification with forcing selecting u.a.r. K non-zeros component from all D components"""
        # E[|C(x)|^2]=(d*d)/(k*k) * E[sum(zi*ei*xi)^2)] = (d*d)/(k*k) * k/d *|x|^2 = d/k * (x^2) = (w + 1) (x^2)
        #  => w = d/k-1
        self.compressorType = CompressorType.RANDK_COMPRESSOR
        self.K = K
        self.D = D
        self.w = self.D / self.K - 1.0
        self.resetStats()

    def makeTopKCompressor(self, D, K):
        """
        Make Top-K sparsification with forcing selecting maximum K component from all D components
        in terms of absolute value.
        """
        # E[|C(x)-x|^2]=(1-a)|x|^2
        self.compressorType = CompressorType.TOPK_COMPRESSOR
        self.K = K
        self.D = D
        self.alpha = self.K / self.D
        self.resetStats()

    def makeRankKCompressor(self, D, K):
        """
        Make Rank-K sparsification compressors with forcing selecting K Rank-1 matrices in truncated SVD expansion
        of the reshaped "X" into matrix
        """
        # E[|C(x)-x|^2]=(1-a)|x|^2
        self.compressorType = CompressorType.RANK_K_COMPRESSOR
        self.K = K
        self.D = D

        self.A = int(D**0.5)
        self.B = int(D**0.5)

        while self.D % self.A != 0:
            self.A = self.A + 1

        self.B = self.D // self.A

        self.alpha = self.K / min(self.A, self.B)

        self.resetStats()

    def makeNaturalCompressorFP32(self, D):
        """Create Natural compressor"""
        self.compressorType = CompressorType.NATURAL_COMPRESSOR_FP32
        self.D = D
        self.w = 1.0 / 8.0
        self.resetStats()

    def getW(self):
        """Get 'w' parameter of unbiased compressor"""
        return self.w

    def getAlphaContraction(self):
        """Get 'alpha' parameter of biased contraction compressor (e.g. top-k)"""
        return self.alpha

    def isContractionCompressor(self):
        """Check that compressor is contraction compressor"""
        return hasattr(self, "alpha")

    def isUnbiasedCompressor(self):
        """Check that compressor is unbiased randomized mapping"""
        return hasattr(self, "w")

    def generateCompressPattern(self, rndgen, device, clientId, H):
        """Generate compress pattern. Sampling stochasticity for each compressors happens only on that part"""
        # For debug
        # print("w for compressor: ", self.getW())

        if self.compressorType == CompressorType.IDENTICAL:
            pass
        elif self.compressorType == CompressorType.LAZY_COMPRESSOR:
            self.testp = rndgen.random()
        elif self.compressorType == CompressorType.RANDK_COMPRESSOR:
            self.S = torch.from_numpy(rndgen.choice(self.D, self.K, replace=False)).to(torch.long).to(device=device)
        elif self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP32:
            self.testp = torch.from_numpy(rndgen.rand(self.D))
        elif self.compressorType == CompressorType.STANDARD_DITHERING_FP32:
            self.testp = torch.from_numpy(rndgen.rand(self.D))
        elif self.compressorType == CompressorType.NATURAL_DITHERING_FP32:
            self.testp = torch.from_numpy(rndgen.rand(self.D))
        elif self.compressorType == CompressorType.TOPK_COMPRESSOR:
            pass
        elif self.compressorType == CompressorType.RANK_K_COMPRESSOR:
            pass

    def compressVector(self, x):
        """ Compress input vector 'x' and produce compressed results """
        d = max(x.shape)
        out = None

        self.last_input_advance = d
        self.last_need_to_send_advance = 0

        # ==============================================================================================================
        if self.compressorType == CompressorType.IDENTICAL:
            out = x
            self.last_need_to_send_advance = d

        elif self.compressorType == CompressorType.LAZY_COMPRESSOR:
            testp = self.testp
            if testp < self.P:
                out = x / self.P
                self.last_need_to_send_advance = d
            else:
                out = torch.zeros_like(x)
                self.last_need_to_send_advance = 0

        elif self.compressorType == CompressorType.RANDK_COMPRESSOR:
            S = self.S
            out = torch.zeros_like(x)
            out[S] = (self.D / self.K) * x[S]
            # We assume that we don't need to send indices
            self.last_need_to_send_advance = self.K

        elif self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP32:
            out = torch.zeros_like(x)

            sign = torch.sign(x)
            alpha = torch.log2(torch.abs(x))
            alpha_down = torch.floor(alpha)
            alpha_up = torch.ceil(alpha)

            pt = (torch.pow(2, alpha_up) - torch.abs(x)) / torch.pow(2, alpha_down)

            if self.testp.device != x.device:
                self.testp = self.testp.to(device=x.device)

            down = (self.testp < pt)

            out[down] = (sign * torch.pow(2, alpha_down))[down]  # with probability pt round to down
            out[~down] = (sign * torch.pow(2, alpha_up))[~down]   # with probability 1-pt round to up
            out[x == 0.0] = 0.0                                   # special rule for handling zero items

            # 8-bit in exponent and extra bit of sign. at hardware level it's possible to get rid of mantissa
            # (with stochastic exp. rouding)
            self.last_need_to_send_advance = 9.0 / 32.0 * d

        elif self.compressorType == CompressorType.STANDARD_DITHERING_FP32:
            out = torch.zeros_like(x)
            pnorm = torch.norm(x, p=self.p)
            pnorm_to_send = pnorm

            sign = torch.sign(x)
            y = torch.abs(x) / pnorm

            if self.testp.device != x.device:
                self.testp = self.testp.to(device=x.device)

            if self.levelsValues.device != x.device:
                self.levelsValues = self.levelsValues.to(device=x.device)

            for s in range(len(self.levelsValues) - 1):
                cond_1 = (y >= self.levelsValues[s])
                cond_2 = (y <= self.levelsValues[s + 1])
                p = (y - self.levelsValues[s + 1]) / (self.levelsValues[s] - self.levelsValues[s + 1])
                cond_3 = (self.testp < p)

                out[cond_1 & cond_2 & cond_3] = self.levelsValues[s]
                out[cond_1 & cond_2 & (~cond_3)] = self.levelsValues[s + 1]

            # special rule for handling zero items
            out[x == 0.0] = 0.0

            out = out * sign * pnorm
            # 1 bit for sign log2(levels) bits to send level
            # 1 scalar for p-norm
            self.last_need_to_send_advance = 1.0 + d * (1.0 + math.ceil(math.log2(self.s))) / 32.0

        elif self.compressorType == CompressorType.NATURAL_DITHERING_FP32:
            out = torch.zeros_like(x)
            pnorm = torch.norm(x, p=self.p)
            pnorm_to_send = pnorm

            sign = torch.sign(x)
            y = torch.abs(x) / pnorm

            if self.testp.device != x.device:
                self.testp = self.testp.to(device=x.device)

            if self.levelsValues.device != x.device:
                self.levelsValues = self.levelsValues.to(device=x.device)

            for s in range(len(self.levelsValues) - 1):
                cond_1 = (y >= self.levelsValues[s])
                cond_2 = (y <= self.levelsValues[s+1])
                p = (y - self.levelsValues[s + 1]) / (self.levelsValues[s] - self.levelsValues[s + 1])
                cond_3 = (self.testp < p)
                out[cond_1 & cond_2 & cond_3] = self.levelsValues[s]
                out[cond_1 & cond_2 & (~cond_3)] = self.levelsValues[s+1]

            # special rule for handling zero items
            out[x == 0.0] = 0.0

            out = y * sign * pnorm
            # 1 bit for sign log2(levels) bits to send level
            # 1 scalar for p-norm
            self.last_need_to_send_advance = 1.0 + d * (1.0 + math.ceil(math.log2(self.s))) / 32.0
        elif self.compressorType == CompressorType.TOPK_COMPRESSOR:
            out = torch.zeros_like(x)
            _, ind = torch.topk(x.abs(), self.K)
            out[ind] = x[ind]
            # Similar to RAND-K we assume that we don't need to send indices
            self.last_need_to_send_advance = self.K
        elif self.compressorType == CompressorType.RANK_K_COMPRESSOR:
            xMat = x.view(self.A, self.B)
            # ==========================================================================================================
            try:
                U, S, Vt = torch.linalg.svd(xMat, full_matrices=False)
            except RuntimeError:
                # Backup plan to make SVD in CPU (sometimes SVD in GPU does not work)
                # Issue: https://github.com/pytorch/pytorch/issues/28293
                U, S, Vt = torch.linalg.svd(xMat.cpu(), full_matrices=False)
                U = U.to(device=xMat.device)
                S = S.to(device=xMat.device)
                Vt = Vt.to(device=xMat.device)
            # ==========================================================================================================

            K = self.K
            K = min(len(S), K)

            Uk = U[..., 0:K]
            Sk = S[0:K]
            Vtk = Vt[0:K, ...]

            out = Uk @ torch.diag(Sk) @ Vtk
            out = out.view(self.D)

            assert Uk.size(0) == self.A
            assert Vtk.size(1) == self.B

            # RANK-K only sends in principle only parameters of dyadic expansion
            self.last_need_to_send_advance = K * (self.A + self.B)

        # ==============================================================================================================
        self.really_need_to_send_components += self.last_need_to_send_advance
        self.total_input_components += self.last_input_advance

        # print("----", (out - x).norm() / x.norm() )
        return out


class ComposedCompressor:
    """Helper class targeted to construct composed compressor from several compressors """
    def __init__(self, c1, c2):
        self.w = 0.0
        self.c1 = c1
        self.c2 = c2

    def resetStats(self):
        self.c1.resetStats()
        self.c2.resetStats()

    def fullName(self):
        self.c1.fullName() + " ( " + self.c2.fullName() + ")"

    def getW(self):
        return (self.c1.getW() + 1) * (self.c2.getW() + 1) - 1

    def compressVector(self, x):
        return self.c1.compressVector(self.c2.compressVector(x))


class ProbabilisticSwitchingCompressor:
    """Probability switching compressor"""
    def __init__(self):
        self.p = []
        self.c = []
        self.p_sum = 0.0

    def addCompressor(self, ci, pi):
        self.c.append(ci)
        self.p.append(pi)
        self.p_sum += pi

    def resetStats(self):
        for ci in self.c:
            ci.resetStats()

    def fullName(self):
        return f"probabilistic switching between {len(self.c)} compressors"

    def getW(self):
        w = 0.0
        for i in range(len(self.c)):
            ci = self.c[i]
            pi = self.p[i]
            w += pi / self.p_sum * ci.getW()

        return w

    def compressVector(self, x, rndgen):
        dice = rndgen.random()
        pTotal = 0.0

        for i in range(len(self.c)):
            pi = self.p[i]
            if dice >= pTotal and dice <= pTotal + pi:
                return self.c[i].compressVector(x)
            pTotal += pi
        return None


def initCompressor(compressorCmdLine, D):
    params = compressorCmdLine.split(":")
    c = Compressor()
    if params[0] == "ident":
        c.makeIdenticalCompressor()
    elif params[0] == "randk":
        if params[1].find("%") == -1:
            K = float(params[1])
            c.makeRandKCompressor(D, math.ceil(K))
        else:
            K = float(params[1][0:-1])/100.0
            c.makeRandKCompressor(D, math.ceil(K * D))
    elif params[0] == "bernulli":
        p = float(params[1])
        c.makeLazyCompressor(p)
    elif params[0] == "natural":
        c.makeNaturalCompressorFP32(D)
    elif params[0] == "qsgd":
        L = int(params[1])
        c.makeQSGD_FP32(D, L)
    elif params[0] == "nat.dithering":
        L = int(params[1])
        pnorm = math.inf
        if len(params) == 3:
            if params[2].lower() == "inf":
                pnorm = math.inf
            else:
                pnorm = int(params[2])

        c.makeNaturalDitheringFP32(D, L, pnorm)
    elif params[0] == "std.dithering":
        L = int(params[1])
        pnorm = math.inf
        if len(params) == 3:
            if params[2].lower() == "inf":
                pnorm = math.inf
            else:
                pnorm = int(params[2])
 
        c.makeStandardDitheringFP32(D, L, pnorm)
    elif params[0] == "topk":
        if params[1].find("%") == -1:
            K = float(params[1])
            c.makeTopKCompressor(D, math.ceil(K))
        else:
            K = float(params[1][0:-1])/100.0
            c.makeTopKCompressor(D, math.ceil(K * D))
    elif params[0] == "rank_k":
        if params[1].find("%") == -1:
            K = float(params[1])
            c.makeRankKCompressor(D, math.ceil(K))
        else:
            K = float(params[1][0:-1])/100.0
            c.makeRankKCompressor(D, math.ceil(K * D))
    elif params[0] == "terngrad":
        c.makeTernGrad(D)
    else:
        assert(not f"Unknown compressor format")

    return c


def test_unbiasedness():
    # python -m pytest compressors.py
    gen = np.random.RandomState()  # Thread specific numpy random generator

    for compressor in ["ident", "randk:10%", "bernulli:0.5", "natural",
                       "qsgd:10", "nat.dithering:10:2", "std.dithering:10:2"]:
        d = 10000
        c = initCompressor(compressor, d)
        x = torch.rand(d)
        x_out = torch.zeros(d)
        for i in range(1000):
            c.generateCompressPattern(gen, "cpu", -1, None)
            x_out += c.compressVector(x)
        x_out /= 1000

        assert (x_out - x).norm() / x.norm() < 0.1


def test_topk_compressor():
    c = initCompressor("topk:50%", 8)
    x_in = torch.Tensor([1, 2, 3, 4, 5, 6, 7, -8])

    rnd = np.random.RandomState()
    c.generateCompressPattern(rnd, x_in.device, -1, None)
    x_out = c.compressVector(x_in)
    print(x_out)
    assert (x_out - torch.Tensor([0, 0, 0, 0, 5, 6, 7, -8], device=x_in.device)).norm() < 0.1


def test_rankk_compressor():
    c = initCompressor("rank_k:100%", 8)
    x_in = torch.Tensor([1, 2, 3, 4, 5, 6, 7, -8])

    rnd = np.random.RandomState()
    c.generateCompressPattern(rnd, x_in.device, -1, None)
    x_out = c.compressVector(x_in)
    print(x_out)
    assert (x_out - x_in).norm().item() < 0.0001
