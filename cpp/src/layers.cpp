#include "layers.h"
#include <cmath>
#include <numeric>

// --- Mask Generation ---
std::tuple<torch::Tensor, std::vector<int64_t>> get_mask(int64_t input_size, std::vector<int64_t> window_size, int64_t inner_size, torch::Device device) {
    std::vector<int64_t> all_size;
    all_size.push_back(input_size);
    for (size_t i = 0; i < window_size.size(); ++i) {
        int64_t layer_size = std::floor(all_size[i] / window_size[i]);
        all_size.push_back(layer_size);
    }

    int64_t seq_length = std::accumulate(all_size.begin(), all_size.end(), 0);
    auto mask = torch::zeros({seq_length, seq_length}, torch::TensorOptions().device(device));

    // Intra-scale mask
    int64_t inner_window = inner_size / 2;
    for (size_t layer_idx = 0; layer_idx < all_size.size(); ++layer_idx) {
        int64_t start = std::accumulate(all_size.begin(), all_size.begin() + layer_idx, 0);
        for (int64_t i = start; i < start + all_size[layer_idx]; ++i) {
            int64_t left_side = std::max(i - inner_window, start);
            int64_t right_side = std::min(i + inner_window + 1, start + all_size[layer_idx]);
            mask.index({i, torch::indexing::Slice(left_side, right_side)}) = 1;
        }
    }

    // Inter-scale mask
    for (size_t layer_idx = 1; layer_idx < all_size.size(); ++layer_idx) {
        int64_t start = std::accumulate(all_size.begin(), all_size.begin() + layer_idx, 0);
        for (int64_t i = start; i < start + all_size[layer_idx]; ++i) {
            int64_t left_side = (start - all_size[layer_idx - 1]) + (i - start) * window_size[layer_idx - 1];
            int64_t right_side;
            if (i == (start + all_size[layer_idx] - 1)) {
                right_side = start;
            } else {
                right_side = (start - all_size[layer_idx - 1]) + (i - start + 1) * window_size[layer_idx - 1];
            }
            mask.index({i, torch::indexing::Slice(left_side, right_side)}) = 1;
            mask.index({torch::indexing::Slice(left_side, right_side), i}) = 1;
        }
    }

    mask = (1 - mask).to(torch::kBool);
    return {mask, all_size};
}

torch::Tensor get_subsequent_mask(int64_t input_size, std::vector<int64_t> window_size, int64_t predict_step, bool truncate) {
    if (truncate) {
        auto mask = torch::zeros({predict_step, input_size + predict_step});
        for (int64_t i = 0; i < predict_step; ++i) {
            mask.index({i, torch::indexing::Slice(0, input_size + i + 1)}) = 1;
        }
        return (1 - mask).to(torch::kBool).unsqueeze(0);
    } else {
        std::vector<int64_t> all_size;
        all_size.push_back(input_size);
        for (size_t i = 0; i < window_size.size(); ++i) {
            int64_t layer_size = std::floor(all_size[i] / window_size[i]);
            all_size.push_back(layer_size);
        }
        int64_t total_size = std::accumulate(all_size.begin(), all_size.end(), 0);
        auto mask = torch::zeros({predict_step, total_size + predict_step});
        for (int64_t i = 0; i < predict_step; ++i) {
            mask.index({i, torch::indexing::Slice(0, total_size + i + 1)}) = 1;
        }
        return (1 - mask).to(torch::kBool).unsqueeze(0);
    }
}

// --- ScaledDotProductAttention ---
ScaledDotProductAttentionImpl::ScaledDotProductAttentionImpl(double temperature, double attn_dropout)
    : temperature_(temperature), dropout_(register_module("dropout", torch::nn::Dropout(attn_dropout))) {}

std::tuple<torch::Tensor, torch::Tensor> ScaledDotProductAttentionImpl::forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask) {
    auto attn = torch::matmul(q / temperature_, k.transpose(2, 3));

    if (mask.defined()) {
        attn = attn.masked_fill(mask, -1e4);
    }

    attn = dropout_->forward(torch::softmax(attn, -1));
    auto output = torch::matmul(attn, v);
    return {output, attn};
}

// --- MultiHeadAttention ---
MultiHeadAttentionImpl::MultiHeadAttentionImpl(int64_t n_head, int64_t d_model, int64_t d_k, int64_t d_v, double dropout, bool normalize_before)
    : n_head_(n_head), d_k_(d_k), d_v_(d_v), normalize_before_(normalize_before),
      w_qs_(register_module("w_qs", torch::nn::Linear(torch::nn::LinearOptions(d_model, n_head * d_k).bias(false)))),
      w_ks_(register_module("w_ks", torch::nn::Linear(torch::nn::LinearOptions(d_model, n_head * d_k).bias(false)))),
      w_vs_(register_module("w_vs", torch::nn::Linear(torch::nn::LinearOptions(d_model, n_head * d_v).bias(false)))),
      fc_(register_module("fc", torch::nn::Linear(n_head * d_v, d_model))),
      attention_(register_module("attention", ScaledDotProductAttention(std::pow(d_k, 0.5), dropout))),
      layer_norm_(register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}).eps(1e-6)))),
      dropout_(register_module("dropout", torch::nn::Dropout(dropout))) {

    torch::nn::init::xavier_uniform_(w_qs_->weight);
    torch::nn::init::xavier_uniform_(w_ks_->weight);
    torch::nn::init::xavier_uniform_(w_vs_->weight);
    torch::nn::init::xavier_uniform_(fc_->weight);
}

std::tuple<torch::Tensor, torch::Tensor> MultiHeadAttentionImpl::forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask) {
    int64_t sz_b = q.size(0);
    int64_t len_q = q.size(1);
    int64_t len_k = k.size(1);
    int64_t len_v = v.size(1);

    auto residual = q;
    if (normalize_before_) {
        q = layer_norm_->forward(q);
    }

    // b x lq x (n*dk) -> b x lq x n x dk
    auto q_proj = w_qs_->forward(q).view({sz_b, len_q, n_head_, d_k_});
    auto k_proj = w_ks_->forward(k).view({sz_b, len_k, n_head_, d_k_});
    auto v_proj = w_vs_->forward(v).view({sz_b, len_v, n_head_, d_v_});

    // b x n x lq x dk
    q_proj = q_proj.transpose(1, 2);
    k_proj = k_proj.transpose(1, 2);
    v_proj = v_proj.transpose(1, 2);

    if (mask.defined()) {
        if (mask.dim() == 3) {
            mask = mask.unsqueeze(1); // broadcasting
        }
    }

    auto output_tuple = attention_->forward(q_proj, k_proj, v_proj, mask);
    auto output = std::get<0>(output_tuple);
    auto attn = std::get<1>(output_tuple);

    // b x lq x n x dv -> b x lq x (n*dv)
    output = output.transpose(1, 2).contiguous().view({sz_b, len_q, -1});
    output = dropout_->forward(fc_->forward(output));
    output += residual;

    if (!normalize_before_) {
        output = layer_norm_->forward(output);
    }

    return {output, attn};
}

// --- PositionwiseFeedForward ---
PositionwiseFeedForwardImpl::PositionwiseFeedForwardImpl(int64_t d_in, int64_t d_hid, double dropout, bool normalize_before)
    : normalize_before_(normalize_before),
      w_1_(register_module("w_1", torch::nn::Linear(d_in, d_hid))),
      w_2_(register_module("w_2", torch::nn::Linear(d_hid, d_in))),
      layer_norm_(register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_in}).eps(1e-6)))),
      dropout_(register_module("dropout", torch::nn::Dropout(dropout))) {}

torch::Tensor PositionwiseFeedForwardImpl::forward(torch::Tensor x) {
    auto residual = x;
    if (normalize_before_) {
        x = layer_norm_->forward(x);
    }
    x = torch::gelu(w_1_->forward(x));
    x = dropout_->forward(x);
    x = w_2_->forward(x);
    x = dropout_->forward(x);
    x += residual;

    if (!normalize_before_) {
        x = layer_norm_->forward(x);
    }
    return x;
}

// --- EncoderLayer ---
EncoderLayerImpl::EncoderLayerImpl(int64_t d_model, int64_t d_inner, int64_t n_head, int64_t d_k, int64_t d_v, double dropout, bool normalize_before)
    : slf_attn_(register_module("slf_attn", MultiHeadAttention(n_head, d_model, d_k, d_v, dropout, normalize_before))),
      pos_ffn_(register_module("pos_ffn", PositionwiseFeedForward(d_model, d_inner, dropout, normalize_before))) {}

std::tuple<torch::Tensor, torch::Tensor> EncoderLayerImpl::forward(torch::Tensor enc_input, torch::Tensor slf_attn_mask) {
    auto output_tuple = slf_attn_->forward(enc_input, enc_input, enc_input, slf_attn_mask);
    auto enc_output = std::get<0>(output_tuple);
    auto enc_slf_attn = std::get<1>(output_tuple);
    
    enc_output = pos_ffn_->forward(enc_output);

    return {enc_output, enc_slf_attn};
}

// --- DecoderLayer ---
DecoderLayerImpl::DecoderLayerImpl(int64_t d_model, int64_t d_inner, int64_t n_head, int64_t d_k, int64_t d_v, double dropout, bool normalize_before)
    : slf_attn_(register_module("slf_attn", MultiHeadAttention(n_head, d_model, d_k, d_v, dropout, normalize_before))),
      pos_ffn_(register_module("pos_ffn", PositionwiseFeedForward(d_model, d_inner, dropout, normalize_before))) {}

std::tuple<torch::Tensor, torch::Tensor> DecoderLayerImpl::forward(torch::Tensor dec_input, torch::Tensor enc_output, torch::Tensor slf_attn_mask) {
    // In Python DecoderLayer:
    // enc_output, enc_slf_attn = self.slf_attn(Q, K, V, mask=slf_attn_mask)
    // Here Q=dec_input, K=enc_output, V=enc_output
    // WAIT: In Python `DecoderLayer.forward(self, Q, K, V, slf_attn_mask)`
    // And in `Decoder.forward`: `dec_enc, _ = self.layers[1](dec_enc, refer_enc, refer_enc, slf_attn_mask=mask)`
    // So Q is dec_input, K is refer_enc, V is refer_enc
    
    auto output_tuple = slf_attn_->forward(dec_input, enc_output, enc_output, slf_attn_mask);
    auto output = std::get<0>(output_tuple);
    auto attn = std::get<1>(output_tuple);

    output = pos_ffn_->forward(output);

    return {output, attn};
}

// --- ConvLayer ---
ConvLayerImpl::ConvLayerImpl(int64_t c_in, int64_t window_size)
    : downConv_(register_module("downConv", torch::nn::Conv1d(torch::nn::Conv1dOptions(c_in, c_in, window_size).stride(window_size)))),
      norm_(register_module("norm", torch::nn::BatchNorm1d(c_in))),
      activation_(register_module("activation", torch::nn::ELU())) {}

torch::Tensor ConvLayerImpl::forward(torch::Tensor x) {
    x = downConv_->forward(x);
    x = norm_->forward(x);
    x = activation_->forward(x);
    return x;
}

// --- Bottleneck_Construct ---
Bottleneck_ConstructImpl::Bottleneck_ConstructImpl(int64_t d_model, std::vector<int64_t> window_size, int64_t d_inner)
    : up_(register_module("up", torch::nn::Linear(d_inner, d_model))),
      down_(register_module("down", torch::nn::Linear(d_model, d_inner))),
      norm_(register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})))) {
    
    for (size_t i = 0; i < window_size.size(); ++i) {
        conv_layers_->push_back(ConvLayer(d_inner, window_size[i]));
    }
    register_module("conv_layers", conv_layers_);
}

torch::Tensor Bottleneck_ConstructImpl::forward(torch::Tensor enc_input) {
    auto temp_input = down_->forward(enc_input).permute({0, 2, 1});
    std::vector<torch::Tensor> all_inputs;
    
    for (int i = 0; i < conv_layers_->size(); ++i) {
        auto& layer = conv_layers_->at<ConvLayerImpl>(i);
        temp_input = layer.forward(temp_input);
        all_inputs.push_back(temp_input);
    }
    
    // torch.cat(all_inputs, dim=2).transpose(1, 2)
    auto combined = torch::cat(all_inputs, 2).transpose(1, 2);
    combined = up_->forward(combined);
    
    auto out = torch::cat({enc_input, combined}, 1);
    out = norm_->forward(out);
    
    return out;
}
