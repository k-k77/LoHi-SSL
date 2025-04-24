import torch
import torch.nn as nn
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask
    
    # def select_sim_negative(self, sim, batch_size):
    #     sim_matric = sim
    #     value = float("-inf")
    #     s = float("+inf")
    #     for i in range(batch_size):
    #         for j in range(batch_size):
    #             if i == j:
    #                 continue
    #             else:
    #                 if sim_matric[i, j] > s:
    #                     sim_matric[i, j] = value
    #                     sim_matric[i, j + batch_size] = value
    #                     sim_matric[j, i] = value
    #                     sim_matric[j, i + batch_size] = value
    #                     sim_matric[i + batch_size, j] = value
    #                     sim_matric[i + batch_size, j + batch_size] = value
    #                     sim_matric[j + batch_size, i] = value
    #                     sim_matric[j + batch_size, i + batch_size] = value
    #     return sim_matric

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        # sim = torch.tensor(cosine_similarity(z.detach().numpy())) / self.temperature
        sim = torch.matmul(z, z.T) / self.temperature
        # np.savetxt('../../data/BRCA/sim_matric.txt', sim.detach().numpy())
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # selected_sim = self.select_sim_negative(sim, self.batch_size)
        # negative_samples = selected_sim[self.mask].reshape(N, -1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

