# coding: utf-8

import torch
import torch.nn as nn

class FairnessLoss(nn.Module):
    def __init__(self, alpha=1.0, reference_position=5):
        super(FairnessLoss, self).__init__()
        self.alpha = alpha
        self.reference_position = reference_position
        self.eps = 1e-7

    def calculate_pairwise_probabilities(self, predictions):
        # Pairwise difference matrix
        """This gives a matrix diff_matrix[i][j] = predictions[i] - predictions[j], 
        representing how much each prediction differs from the others."""
        diff_matrix = predictions.unsqueeze(2) - predictions.unsqueeze(1)
        
        # Sigmoid to get probabilities
        pred_probs = torch.sigmoid(self.alpha * diff_matrix)
        return pred_probs

    def calculate_oracle_probabilities(self, targets):
        # Pairwise difference matrix
        """diff_matrix[i][j] = targets[i] - targets[j]"""
        
        diff_matrix = targets.unsqueeze(2) - targets.unsqueeze(1)
        
        # Oracle probabilities based on target ranking order
        oracle_probs = (diff_matrix > 0).float()
        return oracle_probs

    def forward(self, predictions, targets):
        #print("prediction: ", predictions)
        #print("target: " , targets)
        # Calculate pairwise probabilities for predictions and targets
        pred_probs = self.calculate_pairwise_probabilities(predictions)
        #print("pred_probs: ",pred_probs)

        oracle_probs = self.calculate_oracle_probabilities(targets)
        #print("oracle_probs: ", oracle_probs)
        
        # Masking the reference position
        n_positions = predictions.size(1)
        mask = torch.ones((n_positions, n_positions), device=predictions.device)
        mask[self.reference_position] = 0  # Masking reference row
        mask[:, self.reference_position] = 0  # Masking reference column
        mask.fill_diagonal_(0)
        mask = mask.bool()

        # Compute the binary cross-entropy loss
        pred_probs = torch.clamp(pred_probs, self.eps, 1 - self.eps)
        loss = -(oracle_probs * torch.log(pred_probs) + (1 - oracle_probs) * torch.log(1 - pred_probs))

        # Apply mask and normalize
        loss = loss * mask.unsqueeze(0)
        loss = loss.sum() / (mask.sum() * predictions.size(0))

        return loss
