# ARCHIVE FILE

import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout

class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', pretrained_emb_path="", emb_path=None, proj_dim=None, hidden_size=None, num_layers=None, use_semantic_output=False):
        # emb_path is a dummy param to deal with the current data_config design. Will need to remove. Ignore for now
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.proj_dim = proj_dim
        self.emb_type = emb_type
        self.use_semantic_output = use_semantic_output


        print("We have the write package : DKT model init... ")     
        pretrained_loaded = False
        pretrained_weight = None
        if emb_type.startswith("qid"):
            if pretrained_emb_path.endswith('.pt'):
                emb_w = torch.load(pretrained_emb_path)
                self.emb_size = emb_w.shape[-1]
                self.interaction_emb = Embedding.from_pretrained(emb_w, freeze=True)
                pretrained_loaded = True
                pretrained_weight = emb_w
            else:
                self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        if self.proj_dim is None:
            self.proj_dim = self.emb_size

        self.projection_layer = Linear(self.emb_size, self.proj_dim)

        if hidden_size is None:
            hidden_size = self.proj_dim
        if num_layers is None:
            num_layers = 1

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_layer = LSTM(self.proj_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        self.semantic_output_layer = None
        self.register_buffer("projected_question_embeddings", None)

        if self.use_semantic_output and pretrained_loaded:
            self._build_semantic_output_head(pretrained_weight)
        else:
            self.use_semantic_output = False
        

    def forward(self, q, r):
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
        xproj = self.projection_layer(xemb)
        h, _ = self.lstm_layer(xproj)
        h = self.dropout_layer(h)
        if self.use_semantic_output:
            proj_h = self.semantic_output_layer(h)
            logits = torch.einsum("btd,qd->btq", proj_h, self.projected_question_embeddings)
        else:
            logits = self.out_layer(h)
        y = torch.sigmoid(logits)

        return y

    def _build_semantic_output_head(self, pretrained_weight):
        incorrect = pretrained_weight[:self.num_c]
        correct = pretrained_weight[self.num_c:self.num_c * 2]
        question_embeddings = (incorrect + correct) / 2.0
        with torch.no_grad():
            projected_questions = self.projection_layer(question_embeddings).detach()
        self.projected_question_embeddings = projected_questions
        self.semantic_output_layer = Linear(self.hidden_size, self.proj_dim)
