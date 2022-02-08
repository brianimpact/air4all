import torch
import torch.nn as nn
import torch.nn.functional as F

from model import text_feature_propagation, layernorm_gru


class HiAGM(nn.Module):
    def __init__(self, config, label_ids):
        super(HiAGM, self).__init__()
        self.config = config
        self.label_ids = label_ids
        # TEXT ENCODER (BERT-BASED LANGUAGE MODEL)
        self.text_embedder = torch.hub.load('huggingface/pytorch-transformers', 'model', self.config.model.embedding.type)
        for p in self.text_embedder.parameters():
            p.requires_grad = False
        # EMBEDDING LAYER THAT CONNECTS TEXT ENCODING AND RECURRENT LAYERS
        if (self.config.model.embedding.dimension != self.config.model.rnn.in_dimension) or self.config.model.embedding.additional_layer:
            self.embedding_layer = nn.Linear(
                in_features=self.config.model.embedding.dimension,
                out_features=self.config.model.rnn.in_dimension,
                bias = False
            )
        else:
            self.embedding_layer = nn.Identity()
        self.embedding_dropout = nn.Dropout(self.config.model.embedding.dropout)
        # RECURRENT LAYERS FOR TEXT ENCODING
        if self.config.rnn.layernorm:
            self.text_encoder_rnn = layernorm_gru.LayerNormGRU(
                input_size=self.config.model.rnn.in_dimension,
                hidden_size=self.config.model.rnn.out_dimension,
                num_layers=self.config.model.rnn.layers,
                bidirectional=self.config.model.rnn.bidirectional,
                batch_first=True
            )
        else:
            self.text_encoder_rnn = nn.GRU(
                input_size=self.config.model.rnn.in_dimension,
                hidden_size=self.config.model.rnn.out_dimension,
                num_layers=self.config.model.rnn.layers,
                bidirectional=self.config.model.rnn.bidirectional,
                batch_first=True
            )
        self.rnn_dropout= nn.Dropout(p=self.config.model.rnn.dropout)
        # CONVOLUTIONAL LAYERS WITH TOP K POOLING FOR TEXT ENCODING
        
        self.text_encoder_cnns = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        for kernel in self.config.model.cnn.kernels:
            self.text_encoder_cnns.append(
                torch.nn.Conv1d(
                    in_channels=self.config.model.rnn.out_dimension * 2 if self.config.model.rnn.bidirectional else self.config.model.rnn.out_dimension,
                    out_channels=self.config.model.cnn.dimension,
                    kernel_size=kernel,
                    padding=kernel // 2
                )
            )
            if self.config.model.cnn.layernorm:
                self.layernorms.append(nn.LayerNorm(self.config.model.cnn.dimension))
        self.k = self.config.model.cnn.pooling_k
        # FEATURE AGGREGATION
        self.information_aggregation = text_feature_propagation.TextFeaturePropagation(self.config, self.label_ids)
        
    def forward(self, batch):
        # TEXT ENCODER
        self.text_embedder.eval()
        with torch.no_grad():
            text_embedding = self.text_embedder(batch[0], batch[1], batch[2])['last_hidden_state'] # B*L*D(BERT)
        embedding = self.embedding_layer(text_embedding) # B*L*D(BERT)
        embedding = self.embedding_dropout(embedding)
        if torch.cuda.is_available():
            lengths = batch[4].cpu()
        else:
            lengths = batch[4]
        # RECURRENT LAYERS
        if self.config.rnn.layernorm:
            rnn_output = self.text_encoder_rnn(embedding, lengths)
        else:
            rnn_input = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
            rnn_output, _ = self.text_encoder_rnn(rnn_input)
            rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True) # B*L*D(RNN)
        rnn_output = rnn_output.transpose(1, 2) # B*D(RNN)*L
        rnn_output = self.rnn_dropout(rnn_output)
        # CONVOLUTIONAL LAYERS
        cnn_output = []
        for i in range(len(self.text_encoder_cnns)):
            out = self.text_encoder_cnns[i](rnn_output)
            if self.config.model.cnn.layernorm:
                out = self.layernorms[i](out.transpose(1, 2)).transpose(1, 2)
            out = torch.topk(out, self.k)[0] # B*kD(CNN)
            out = F.relu(out) # B*kD(CNN)
            cnn_output.append(out.view(out.size(0), -1)) # B*kD(CNN)
        cnn_output = torch.stack(cnn_output, 1) # B*(#CNN)k*D(CNN)
        # FEATURE AGGREGATION (LOGIT CALCULATION)
        logits = self.information_aggregation(cnn_output)

        return logits
