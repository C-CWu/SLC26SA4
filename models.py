import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_peaks_Model(nn.Module):
    def __init__(self, feature_length, hidden_size, num_layer, bidirectional=True):
        super(LSTM_peaks_Model, self).__init__()
        self.lstm = nn.LSTM(feature_length, hidden_size, num_layer, batch_first=True, bidirectional=bidirectional)
        self.age_embedding = nn.Linear(1, hidden_size)
        self.step_embedding = nn.Linear(1, hidden_size)

        def make_decoder():
            return nn.Sequential(
                nn.Linear(hidden_size * 4, 128) if bidirectional else nn.Linear(hidden_size * 3, 128),
                nn.Mish(),
                nn.Linear(128, 64),
                nn.Mish(),
                nn.Linear(64, 32),
                nn.Mish(),
                nn.Linear(32, 1),
            )

        self.decoders = nn.ModuleList([make_decoder() for _ in range(6)])
        self.Mish = nn.Mish()

    def forward(self, x, y_age, step):
        output, _ = self.lstm(x)
        output = self.Mish(output[:, -1, :])
        
        y_age_embedded = self.age_embedding(y_age.unsqueeze(1))
        step_embedded = self.step_embedding(step.unsqueeze(1))

        output = torch.cat((output, y_age_embedded, step_embedded), 1)

        stacked_tensor = torch.cat([decoder(output) for decoder in self.decoders], dim=1)
        return stacked_tensor

class LSTM_6(nn.Module):
    def __init__(self, feature_length, hidden_size, num_layer, bidirectional=True):
        super(LSTM_6, self).__init__()
        
        def make_decoder():
            lstm = nn.LSTM(feature_length, hidden_size, num_layer, batch_first=True, bidirectional=bidirectional)
            additional_features = 2  # y_age and step
            input_size = hidden_size * 2 + additional_features if bidirectional else hidden_size + additional_features
            
            return nn.Sequential(
                lstm,
                nn.Linear(input_size, 128),
                nn.Mish(),
                nn.Linear(128, 64),
                nn.Mish(),
                nn.Linear(64, 32),
                nn.Mish(),
                nn.Linear(32, 1),
            )
        
        self.decoders = nn.ModuleList([make_decoder() for _ in range(6)])
        self.Mish = nn.Mish()

    def forward(self, x, y_age, step):
        results = []
        y_age = y_age.unsqueeze(1)
        step = step.unsqueeze(1)

        for decoder in self.decoders:
            lstm = decoder[0]
            output, _ = lstm(x)
            output = self.Mish(output[:, -1, :])
            output = torch.cat((output, y_age, step), 1)
            for layer in decoder[1:]:
                output = layer(output)
            results.append(output)
        
        stacked_tensor = torch.cat(results, dim=1)
        return stacked_tensor

class AModel(nn.Module):
    def __init__(self, feature_length, hidden_size, num_layer, embedding_dim, bidirectional=True):
        super(AModel, self).__init__()
        self.lstm = nn.LSTM(feature_length, hidden_size, num_layer, batch_first=True, bidirectional=bidirectional)
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.feature_length = feature_length
        self.bidirectional = bidirectional
        self.attention = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1)
        self.Mish = nn.Mish()
        self.age_embedding = nn.Linear(1, embedding_dim)
        self.step_embedding = nn.Linear(1, embedding_dim)

        decoder_output_size = hidden_size * 2 + embedding_dim * 2 if bidirectional else hidden_size + embedding_dim * 2
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(decoder_output_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            ) for _ in range(6)
        ])

    def forward(self, x, y_age, step):
        output, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(output).squeeze(-1), dim=-1)
        context_vector = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)
        context_vector = context_vector + output[:, -1, :]

        y_age_embedded = self.age_embedding(y_age.unsqueeze(1))
        step_embedded = self.step_embedding(step.unsqueeze(1))
        
        # Concatenate additional features
        output = torch.cat((context_vector, y_age_embedded, step_embedded), 1)

        # output = torch.cat((context_vector, y_age, step), 1)

        stacked_tensor = torch.cat([decoder(output) for decoder in self.decoders], dim=1)
        return stacked_tensor

    
class Transformer_peaks_Model(nn.Module):
    def __init__(self, feature_length, hidden_size, num_layers, nhead):
        super(Transformer_peaks_Model, self).__init__()
        self.transformer = nn.Transformer(d_model=feature_length, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True)
        self.age_embedding = nn.Linear(1, hidden_size)
        self.step_embedding = nn.Linear(1, hidden_size)

        def make_decoder():
            return nn.Sequential(
                nn.Linear(feature_length + hidden_size * 2, 128),
                nn.Mish(),
                nn.Linear(128, 64),
                nn.Mish(),
                nn.Linear(64, 32),
                nn.Mish(),
                nn.Linear(32, 1),
            )

        self.decoders = nn.ModuleList([make_decoder() for _ in range(6)])        
        self.Mish = nn.Mish()

    def forward(self, x, y_age, step):
        output = self.transformer(x, x)
        output = self.Mish(output[:, -1, :])
        
        y_age_embedded = self.age_embedding(y_age.unsqueeze(1))
        step_embedded = self.step_embedding(step.unsqueeze(1))

        
        output = torch.cat((output, y_age_embedded, step_embedded), 1)

        stacked_tensor = torch.cat([decoder(output) for decoder in self.decoders], dim=1)
        return stacked_tensor
    
class MLP_peaks_Model(nn.Module):
    def __init__(self, feature_length, r_size, hidden_size):
        super(MLP_peaks_Model, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(r_size * feature_length, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish()
        )
        
        self.age_embedding = nn.Linear(1, hidden_size)
        self.step_embedding = nn.Linear(1, hidden_size)

        def make_decoder():
            return nn.Sequential(
                nn.Linear(hidden_size * 3, 128),
                nn.Mish(),
                nn.Linear(128, 64),
                nn.Mish(),
                nn.Linear(64, 32),
                nn.Mish(),
                nn.Linear(32, 1),
            )

        self.decoders = nn.ModuleList([make_decoder() for _ in range(6)])
        
        self.Mish = nn.Mish()

    def forward(self, x, y_age, step):
        output = x.view(x.size(0), -1)
        output = self.feature_extractor(output)
        
        y_age_embedded = self.age_embedding(y_age.unsqueeze(1))
        step_embedded = self.step_embedding(step.unsqueeze(1))

        output = torch.cat((output, y_age_embedded, step_embedded), 1)

        stacked_tensor = torch.cat([decoder(output) for decoder in self.decoders], dim=1)
        return stacked_tensor
