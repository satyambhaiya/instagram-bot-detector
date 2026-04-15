"""
BotDetectorNet — Instagram Bot Detection
3-class: Human(0) / Bot(1) / Suspicious(2)
Architecture: Residual-style deep MLP with BatchNorm + Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Small residual block: Linear → BN → ReLU → Linear → BN + skip."""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.relu(self.block(x) + x))


class BotDetectorNet(nn.Module):
    """
    Instagram Bot Detector — 3-class classifier.

    Input : 24 Instagram behavioral features
    Output: logits for [Human, Bot, Suspicious]
    """
    INPUT_DIM   = 27
    NUM_CLASSES = 3

    FEATURE_COLS = [
        "followers_count", "following_count", "posts_count",
        "avg_likes_per_post", "avg_comments_per_post",
        "posts_per_day", "follower_following_ratio", "engagement_rate",
        "profile_pic", "bio_length", "has_url_in_bio", "is_verified",
        "is_private", "account_age_days", "username_digit_ratio",
        "username_length", "night_activity_ratio", "avg_caption_length",
        "hashtags_per_post", "mentions_per_post", "story_frequency",
        "reels_ratio", "comment_reply_rate", "unique_commenters_ratio",
        # v2.1 — engagement quality + behavioral consistency + mass-follow detection
        "likes_comments_ratio",          # avg_likes / avg_comments (high = bought likes)
        "posting_regularity",            # 0 = fixed intervals (bot), 1 = irregular (human)
        "following_to_followers_ratio",  # following / followers (high = mass-follow bot)
    ]
    LABEL_NAMES = ["Human", "Bot", "Suspicious"]

    def __init__(self, input_dim=27, hidden=128, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res1 = ResidualBlock(hidden, dropout)
        self.res2 = ResidualBlock(hidden, dropout)

        self.neck = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
        )
        self.head = nn.Linear(64, self.NUM_CLASSES)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.neck(x)
        return self.head(x)               # raw logits (batch, 3)

    def predict(self, x: torch.Tensor):
        """Returns (predicted_class, probabilities)."""
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=-1)
            cls    = probs.argmax(dim=-1)
        return cls, probs


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for imbalanced classes."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # class weights tensor

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t  = torch.exp(-ce)
        loss = ((1 - p_t) ** self.gamma) * ce
        return loss.mean()


if __name__ == "__main__":
    model = BotDetectorNet()
    print(model)
    x = torch.randn(8, 27)
    cls, probs = model.predict(x)
    print(f"Predicted classes : {cls.tolist()}")
    print(f"Probabilities     :\n{probs.round(decimals=3)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters  : {total:,}")
