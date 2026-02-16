#pragma once

#include <torch/torch.h>
#include "layers.h"
#include "embeddings.h"

struct ModelOptions {
    int64_t input_size;
    int64_t predict_size;
    int64_t d_model;
    int64_t d_inner_hid;
    int64_t d_k;
    int64_t d_v;
    int64_t n_head;
    int64_t n_layer;
    std::vector<int64_t> window_size;
    double dropout;
    int64_t enc_in;
    int64_t inner_size = 3;
    int64_t d_bottleneck = 128;
    torch::Device device = torch::kCPU;
};

// Encoder
class EncoderImpl : public torch::nn::Module {
public:
    EncoderImpl(ModelOptions opt);
    torch::Tensor forward(torch::Tensor x, torch::Tensor t);

private:
    DataEmbedding enc_embedding_ = nullptr;
    Bottleneck_Construct conv_layers_ = nullptr;
    torch::nn::ModuleList layers_ = nullptr;
    torch::Tensor mask_;
    std::vector<int64_t> all_size_;
};
TORCH_MODULE(Encoder);

// Decoder
class DecoderImpl : public torch::nn::Module {
public:
    DecoderImpl(ModelOptions opt, torch::Tensor mask);
    torch::Tensor forward(torch::Tensor x_dec, torch::Tensor x_mark_dec, torch::Tensor refer);

private:
    DataEmbedding dec_embedding_ = nullptr;
    torch::nn::ModuleList layers_ = nullptr;
    torch::Tensor mask_;
};
TORCH_MODULE(Decoder);

// Predictor
class PredictorImpl : public torch::nn::Module {
public:
    PredictorImpl(int64_t dim, int64_t num_types);
    torch::Tensor forward(torch::Tensor data);

private:
    torch::nn::Linear linear_ = nullptr;
};
TORCH_MODULE(Predictor);

// Full Pyraformer Model
class PyraformerImpl : public torch::nn::Module {
public:
    PyraformerImpl(ModelOptions opt);
    torch::Tensor forward(torch::Tensor x_enc, torch::Tensor t_enc, torch::Tensor x_dec, torch::Tensor t_dec, bool pretrain = false);

    // Getters for device handling in training loop
    int64_t input_size() const { return opt_.input_size; }
    int64_t predict_size() const { return opt_.predict_size; }
    ModelOptions opt_;

private:
    Encoder encoder_ = nullptr;
    Decoder decoder_ = nullptr;
    Predictor predictor_ = nullptr;
};
TORCH_MODULE(Pyraformer);
