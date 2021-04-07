import random

import torch

from DA2Lite.compression.pruning.methods.criteria_base import CriteriaBase



class RandomCriteria(CriteriaBase):

    def get_prune_idx(self, weights, pruning_ratio=0.0): 
        if pruning_ratio <= 0: return []

        n = len(weights)
        n_to_prune = int(apruning_ratiomount * n)
        indices = random.sample( list( range(n) ), k=n_to_prune )
        return indices