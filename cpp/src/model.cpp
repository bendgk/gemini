#include "model.h"
#include <iostream>

// --- Encoder ---
EncoderImpl::EncoderImpl(ModelOptions opt)
    : enc_embedding_(register_module("enc_embedding", DataEmbedding(opt.enc_in, opt.d_model, opt.dropout))),
      conv_layers_(register_module("conv_layers", Bottleneck_Construct(opt.d_model, opt.window_size, opt.d_bottleneck))) {
    
    // Get mask
    auto mask_tuple = get_mask(opt.input_size, opt.window_size, opt.inner_size, opt.device);
    mask_ = std::get<0>(mask_tuple);
    // Register mask as buffer so it moves with model
    register_buffer("mask", mask_);
    all_size_ = std::get<1>(mask_tuple);

    // Layers
    for (int i = 0; i < opt.n_layer; ++i) {
        layers_->push_back(EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, opt.dropout, false));
    }
    register_module("layers", layers_);
}

torch::Tensor EncoderImpl::forward(torch::Tensor x, torch::Tensor t) {
    auto seq_enc = enc_embedding_->forward(x, t); // [B, L, D]

    // mask needs to repeat?
    // In python: mask = self.mask.repeat(len(seq_enc), 1, 1).to(x.device)
    // Here mask is [L, L]. 
    // Usually attention takes [B, H, L, L] or [B, 1, L, L].
    // If mask is [L, L], broadcasting should handle it, but maybe we need unsqueeze.
    // Python code repeats it to [B, L, L].
    
    // Check MultiHeadAttention::forward: 
    // if mask.dim() == 3: mask = mask.unsqueeze(1) (for head)
    // So if we pass [B, L, L], it becomes [B, 1, L, L].
    
    auto mask_expanded = mask_.unsqueeze(0).expand({seq_enc.size(0), -1, -1});

    seq_enc = conv_layers_->forward(seq_enc);

    for (int i = 0; i < layers_->size(); ++i) {
        auto& layer = layers_->at<EncoderLayerImpl>(i);
        auto out = layer.forward(seq_enc, mask_expanded);
        seq_enc = std::get<0>(out);
    }
    
    return seq_enc;
}

// --- Decoder ---
DecoderImpl::DecoderImpl(ModelOptions opt, torch::Tensor mask)
    : dec_embedding_(register_module("dec_embedding", DataEmbedding(opt.enc_in, opt.d_model, opt.dropout))),
      mask_(mask) {
    
    // Register mask
    register_buffer("mask", mask_);

    // 2 Layers as per Python code
    layers_->push_back(DecoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, opt.dropout, false));
    layers_->push_back(DecoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, opt.dropout, false));
    register_module("layers", layers_);
}

torch::Tensor DecoderImpl::forward(torch::Tensor x_dec, torch::Tensor x_mark_dec, torch::Tensor refer) {
    auto dec_enc = dec_embedding_->forward(x_dec, x_mark_dec);
    
    // Layer 0
    auto out0 = layers_->at<DecoderLayerImpl>(0).forward(dec_enc, refer);
    dec_enc = std::get<0>(out0);

    auto refer_enc = torch::cat({refer, dec_enc}, 1);
    
    auto mask_expanded = mask_.repeat({(long)dec_enc.size(0), 1, 1}).to(dec_enc.device());
    
    auto out1 = layers_->at<DecoderLayerImpl>(1).forward(dec_enc, refer_enc, mask_expanded);
    dec_enc = std::get<0>(out1);
    
    return dec_enc;
}

// --- Predictor ---
PredictorImpl::PredictorImpl(int64_t dim, int64_t num_types)
    : linear_(register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(dim, num_types).bias(false)))) {
    torch::nn::init::xavier_normal_(linear_->weight);
}

torch::Tensor PredictorImpl::forward(torch::Tensor data) {
    return linear_->forward(data);
}

// --- Pyraformer ---
PyraformerImpl::PyraformerImpl(ModelOptions opt) : opt_(opt), encoder_(nullptr), decoder_(nullptr), predictor_(nullptr) {
    encoder_ = register_module("encoder", Encoder(opt));
    
    auto mask = get_subsequent_mask(opt.input_size, opt.window_size, opt.predict_size, false);
    decoder_ = register_module("decoder", Decoder(opt, mask));
    
    predictor_ = register_module("predictor", Predictor(opt.d_model, opt.enc_in));
}

torch::Tensor PyraformerImpl::forward(torch::Tensor x_enc, torch::Tensor t_enc, torch::Tensor x_dec, torch::Tensor t_dec, bool pretrain) {
    auto enc_output = encoder_->forward(x_enc, t_enc);
    auto dec_enc = decoder_->forward(x_dec, t_dec, enc_output);

    if (pretrain) {
        dec_enc = torch::cat({enc_output.slice(1, 0, opt_.input_size), dec_enc}, 1);
    }
    
    auto pred = predictor_->forward(dec_enc);
    return pred;
}
