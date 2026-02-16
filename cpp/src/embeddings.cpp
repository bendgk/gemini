#include "embeddings.h"
#include <cmath>
#include <iostream>

PositionalEmbeddingImpl::PositionalEmbeddingImpl(int d_model, int max_len) {
    pe_ = torch::zeros({max_len, d_model});
    pe_.set_requires_grad(false);

    auto position = torch::arange(0, max_len, torch::kFloat).unsqueeze(1);
    auto div_term = torch::exp(torch::arange(0, d_model, 2, torch::kFloat) * -(std::log(10000.0) / d_model));

    pe_.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)}, 
                   torch::sin(position * div_term));
    pe_.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)}, 
                   torch::cos(position * div_term));

    pe_ = pe_.unsqueeze(0);
    register_buffer("pe", pe_);
}

torch::Tensor PositionalEmbeddingImpl::forward(torch::Tensor x) {
    return pe_.slice(1, 0, x.size(1));
}

TokenEmbeddingImpl::TokenEmbeddingImpl(int c_in, int d_model) {
    // padding=1 if torch version >= 1.5.0 else 2. PyTorch C++ matches current version.
    // python: padding=1, padding_mode='circular'
    // Conv1d(c_in, d_model, kernel_size=3, padding=1)
    
    // Libtorch Conv1d options
    // Note: padding_mode='circular' is supported in recent libtorch versions via specialized functional call or options?
    // checking torch::nn::Conv1dOptions... it has .padding_mode(torch::kCircular)
    
    conv_ = register_module("conv", torch::nn::Conv1d(
        torch::nn::Conv1dOptions(c_in, d_model, 3).padding(1).padding_mode(torch::kCircular)
    ));

    // Init weights
    for (auto& p : conv_->parameters()) {
        if (p.dim() > 1) {
            torch::nn::init::kaiming_normal_(p, 0, torch::kFanIn, torch::kLeakyReLU);
        }
    }
}

torch::Tensor TokenEmbeddingImpl::forward(torch::Tensor x) {
    // x: [batch, seq_len, c_in]
    // permute to [batch, c_in, seq_len] for Conv1d
    return conv_->forward(x.permute({0, 2, 1})).transpose(1, 2);
}

TemporalEmbeddingImpl::TemporalEmbeddingImpl(int d_model) {
    int d_inp = 4; // seconds, minutes, hours, day
    embed_ = register_module("embed", torch::nn::Linear(d_inp, d_model));
}

torch::Tensor TemporalEmbeddingImpl::forward(torch::Tensor x) {
    return embed_->forward(x);
}

DataEmbeddingImpl::DataEmbeddingImpl(int c_in, int d_model, double dropout, int max_len)
    : token_embedding_(register_module("token_embedding", TokenEmbedding(c_in, d_model))),
      position_embedding_(register_module("position_embedding", PositionalEmbedding(d_model, max_len))),
      temporal_embedding_(register_module("temporal_embedding", TemporalEmbedding(d_model))),
      dropout_(register_module("dropout", torch::nn::Dropout(dropout))) {
}

torch::Tensor DataEmbeddingImpl::forward(torch::Tensor x, torch::Tensor t) {
    // x: features, t: covariates
    auto x_emb = token_embedding_->forward(x) + position_embedding_->forward(x) + temporal_embedding_->forward(t);
    return dropout_->forward(x_emb);
}
