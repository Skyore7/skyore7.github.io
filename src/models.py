import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchedDenses(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(BatchedDenses, self).__init__()
        layers = []
        in_features = input_size
        for out_features in layer_sizes:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class BatchedConv2Ds(nn.Module):
    def __init__(self, input_channels, conv_layers):
        super(BatchedConv2Ds, self).__init__()
        layers = []
        in_channels = input_channels
        for out_channels in conv_layers:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class RocketLeagueModel(nn.Module):
    def __init__(self):
        super(RocketLeagueModel, self).__init__()
        # Ball preprocessing
        self.ball_dense = BatchedDenses(6, [24, 18, 10])

        # Player convolution
        self.player_conv = BatchedConv2Ds(8, [40, 32, 24, 18])

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 + 18 * 2, 81),
            nn.ReLU(),
            nn.Linear(81, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 3)
        )

    def forward(self, inputs):
        # Split inputs into ball and players
        ball = inputs[:, :6]
        playersA = inputs[:, 6:30].reshape(-1, 3, 8)  # [batch_size, num_players, features]
        playersB = inputs[:, 30:54].reshape(-1, 3, 8)  # [batch_size, num_players, features]

        # Process ball
        ball_processed = self.ball_dense(ball)  # [batch_size, ball_features]

        # Process players (shared `player_conv` for each player)
        # Permute to match [batch_size, channels, height, width] = [batch_size, features, num_players, 1]
        playersA_perm = playersA.permute(0, 2, 1).unsqueeze(-1)  # [batch_size, features, num_players, 1]
        playersB_perm = playersB.permute(0, 2, 1).unsqueeze(-1)  # [batch_size, features, num_players, 1]

        # Apply convolution to players
        playersA_conv = self.player_conv(playersA_perm)  # [batch_size, out_channels, num_players, 1]
        playersB_conv = self.player_conv(playersB_perm)  # [batch_size, out_channels, num_players, 1]

        # Aggregate player features (mean pooling across players)
        teamA_agg = playersA_conv.mean(dim=2).squeeze(-1)  # [batch_size, out_channels]
        teamB_agg = playersB_conv.mean(dim=2).squeeze(-1)  # [batch_size, out_channels]

        # Concatenate ball features with aggregated team features
        combined = torch.cat([ball_processed, teamA_agg, teamB_agg], dim=1)  # [batch_size, combined_features]

        # Pass through fully connected layers
        x = self.fc(combined)  # [batch_size, 42]

        # Compute outputs for each target
        outputs = F.softmax(x, dim=1)
        return outputs




# Example usage
""" model = RocketLeagueModel()
inputs = torch.randn(32, 54)  # Batch size of 32, input features 54
outputs = model(inputs)
for i, output in enumerate(outputs):
    print(f"Output {i} shape: {output.shape}")
    print(output)
 """