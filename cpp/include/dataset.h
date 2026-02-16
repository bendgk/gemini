#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

struct GeminiRow {
    uint64_t timestamp;
    double bid;
    double bid_qty;
    double ask;
    double ask_qty;
};

class GeminiDataset : public torch::data::datasets::Dataset<GeminiDataset> {
public:
    GeminiDataset(const std::string& file_path, int input_size = 2048, int predict_size = 1024);

    // Returns a pair of tensors: {features, covariates}
    // features: [seq_len, 7]
    // covariates: [seq_len, 4]
    torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

    // Return the size of the dataset
    torch::optional<size_t> size() const override;

    // Compute mean and std on the first `train_size` samples and normalize data
    void normalize(size_t train_size);

private:
    void load_data(const std::string& file_path);
    void precompute_features();
    void compute_covariates();

    std::vector<GeminiRow> raw_data_;
    
    // features: [N, 7]
    // covariates: [N, 4]
    torch::Tensor features_;
    torch::Tensor covariates_;

    int input_size_;
    int predict_size_;
    int seq_len_;
};
