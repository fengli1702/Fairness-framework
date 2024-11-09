# coding: utf-8
import torch
import torch.nn as nn

class FairnessLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(FairnessLoss, self).__init__()
        self.alpha = alpha
        self.eps = 1e-7

    def calculate_pairwise_probabilities(self, predictions):
        # Compute pairwise differences for predictions
        # diff_matrix[i][j] = predictions[i] - predictions[j],
        diff_matrix = predictions.unsqueeze(2) - predictions.unsqueeze(1)
        # Convert differences to probabilities using sigmoid
        pred_probs = torch.sigmoid(self.alpha * diff_matrix)
        return pred_probs

    def calculate_oracle_probabilities(self, targets):
        # Compute pairwise differences based on the ranking in targets
        diff_matrix = targets.unsqueeze(2) - targets.unsqueeze(1)
        # Generate oracle probabilities: if target[i] < target[j], oracle_probs[i][j] = 1
        oracle_probs = (diff_matrix < 0).float()
        return oracle_probs

    def forward(self, predictions, targets):
        # Calculate the pairwise probabilities for predictions and targets
        pred_probs = self.calculate_pairwise_probabilities(predictions)
        oracle_probs = self.calculate_oracle_probabilities(targets)
        
        # Mask the diagonal (self-comparisons)
        n_positions = predictions.size(1)
        mask = torch.ones((n_positions, n_positions), device=predictions.device).bool()
        mask.fill_diagonal_(0)

        # Binary cross-entropy loss between oracle and predicted probabilities
        pred_probs = torch.clamp(pred_probs, self.eps, 1 - self.eps)
        loss = -(oracle_probs * torch.log(pred_probs) + (1 - oracle_probs) * torch.log(1 - pred_probs))

        # Apply mask and normalize loss
        loss = loss * mask.unsqueeze(0)
        loss = loss.sum() / (mask.sum() * predictions.size(0))
        return loss
