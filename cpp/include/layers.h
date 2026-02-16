#pragma once

#include <torch/torch.h>
#include <vector>

// Mask generation functions
std::tuple<torch::Tensor, std::vector<int64_t>> get_mask(int64_t input_size, std::vector<int64_t> window_size, int64_t inner_size, torch::Device device);
torch::Tensor get_subsequent_mask(int64_t input_size, std::vector<int64_t> window_size, int64_t predict_step, bool truncate);

// Scaled Dot Product Attention
class ScaledDotProductAttentionImpl : public torch::nn::Module {
public:
    ScaledDotProductAttentionImpl(double temperature, double attn_dropout = 0.1);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask = torch::Tensor());

private:
    double temperature_;
    torch::nn::Dropout dropout_;
};
TORCH_MODULE(ScaledDotProductAttention);

// Multi-Head Attention
class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(int64_t n_head, int64_t d_model, int64_t d_k, int64_t d_v, double dropout = 0.1, bool normalize_before = true);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask = torch::Tensor());

private:
    int64_t n_head_;
    int64_t d_k_;
    int64_t d_v_;
    bool normalize_before_;

    torch::nn::Linear w_qs_ = nullptr;
    torch::nn::Linear w_ks_ = nullptr;
    torch::nn::Linear w_vs_ = nullptr;
    torch::nn::Linear fc_ = nullptr;
    ScaledDotProductAttention attention_ = nullptr;
    torch::nn::LayerNorm layer_norm_ = nullptr;
    torch::nn::Dropout dropout_ = nullptr;
};
TORCH_MODULE(MultiHeadAttention);

// Position-wise Feed-Forward
class PositionwiseFeedForwardImpl : public torch::nn::Module {
public:
    PositionwiseFeedForwardImpl(int64_t d_in, int64_t d_hid, double dropout = 0.1, bool normalize_before = true);
    torch::Tensor forward(torch::Tensor x);

private:
    bool normalize_before_;
    torch::nn::Linear w_1_ = nullptr;
    torch::nn::Linear w_2_ = nullptr;
    torch::nn::LayerNorm layer_norm_ = nullptr;
    torch::nn::Dropout dropout_ = nullptr;
};
TORCH_MODULE(PositionwiseFeedForward);

// Encoder Layer
class EncoderLayerImpl : public torch::nn::Module {
public:
    EncoderLayerImpl(int64_t d_model, int64_t d_inner, int64_t n_head, int64_t d_k, int64_t d_v, double dropout = 0.1, bool normalize_before = true);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor enc_input, torch::Tensor slf_attn_mask = torch::Tensor());

private:
    MultiHeadAttention slf_attn_ = nullptr;
    PositionwiseFeedForward pos_ffn_ = nullptr;
};
TORCH_MODULE(EncoderLayer);

// Decoder Layer
class DecoderLayerImpl : public torch::nn::Module {
public:
    DecoderLayerImpl(int64_t d_model, int64_t d_inner, int64_t n_head, int64_t d_k, int64_t d_v, double dropout = 0.1, bool normalize_before = true);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor dec_input, torch::Tensor enc_output, torch::Tensor slf_attn_mask = torch::Tensor());

private:
    MultiHeadAttention slf_attn_ = nullptr;
    PositionwiseFeedForward pos_ffn_ = nullptr;
};
TORCH_MODULE(DecoderLayer);

// Conv Layer
class ConvLayerImpl : public torch::nn::Module {
public:
    ConvLayerImpl(int64_t c_in, int64_t window_size);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv1d downConv_ = nullptr;
    torch::nn::BatchNorm1d norm_ = nullptr;
    torch::nn::ELU activation_ = nullptr;
};
TORCH_MODULE(ConvLayer);

// Bottleneck Construct
class Bottleneck_ConstructImpl : public torch::nn::Module {
public:
    Bottleneck_ConstructImpl(int64_t d_model, std::vector<int64_t> window_size, int64_t d_inner);
    torch::Tensor forward(torch::Tensor enc_input);

private:
    torch::nn::ModuleList conv_layers_ = nullptr;
    torch::nn::Linear up_ = nullptr;
    torch::nn::Linear down_ = nullptr;
    torch::nn::LayerNorm norm_ = nullptr;
};
TORCH_MODULE(Bottleneck_Construct);
