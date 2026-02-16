#pragma once

#include <torch/torch.h>

class PositionalEmbeddingImpl : public torch::nn::Module {
public:
    PositionalEmbeddingImpl(int d_model, int max_len = 5000);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::Tensor pe_;
};
TORCH_MODULE(PositionalEmbedding);

class TokenEmbeddingImpl : public torch::nn::Module {
public:
    TokenEmbeddingImpl(int c_in, int d_model);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv1d conv_ = nullptr;
};
TORCH_MODULE(TokenEmbedding);

class TemporalEmbeddingImpl : public torch::nn::Module {
public:
    TemporalEmbeddingImpl(int d_model);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear embed_ = nullptr;
};
TORCH_MODULE(TemporalEmbedding);

class DataEmbeddingImpl : public torch::nn::Module {
public:
    DataEmbeddingImpl(int c_in, int d_model, double dropout = 0.1, int max_len = 10000);
    torch::Tensor forward(torch::Tensor x, torch::Tensor t);

private:
    TokenEmbedding token_embedding_ = nullptr;
    PositionalEmbedding position_embedding_ = nullptr;
    TemporalEmbedding temporal_embedding_ = nullptr;
    torch::nn::Dropout dropout_ = nullptr;
};
TORCH_MODULE(DataEmbedding);
