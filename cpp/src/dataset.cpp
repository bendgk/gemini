#include "dataset.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <arpa/inet.h> // for ntohl, etc. But checking endianness manualy is safer for 64bit doubles
#include <cstring>
#include <algorithm>

// Endianness helper
// The file is Big Endian (>Qdddd). x86 is Little Endian.
// We need to swap bytes if system is Little Endian.

static uint64_t swap_uint64(uint64_t val) {
    val = ((val << 8) & 0xFF00FF00FF00FF00ULL ) | ((val >> 8) & 0x00FF00FF00FF00FFULL );
    val = ((val << 16) & 0xFFFF0000FFFF0000ULL ) | ((val >> 16) & 0x0000FFFF0000FFFFULL );
    return (val << 32) | (val >> 32);
}

static double swap_double(double val) {
    uint64_t tmp;
    std::memcpy(&tmp, &val, sizeof(uint64_t));
    tmp = swap_uint64(tmp);
    std::memcpy(&val, &tmp, sizeof(uint64_t));
    return val;
}

GeminiDataset::GeminiDataset(const std::string& file_path, int input_size, int predict_size) 
    : input_size_(input_size), predict_size_(predict_size) {
    seq_len_ = input_size_ + predict_size_;
    load_data(file_path);
    precompute_features();
    compute_covariates();
}

void GeminiDataset::load_data(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    // Structure is >Qdddd (40 bytes)
    // Q: uint64 (8 bytes)
    // d: double (8 bytes) * 4
    
    // Check file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_records = file_size / 40;
    if (num_records < (size_t)seq_len_) {
        throw std::runtime_error("File too small");
    }

    raw_data_.reserve(num_records);

    struct RawRow {
        uint64_t ts;
        double bid;
        double bid_qty;
        double ask;
        double ask_qty;
    };

    RawRow row;
    while (file.read(reinterpret_cast<char*>(&row), sizeof(RawRow))) {
        // Swap endianness if necessary (assuming Little Endian host)
        // Check host endianness
        uint16_t number = 0x1;
        char *numPtr = (char*)&number;
        bool isLittleEndian = (numPtr[0] == 1);

        GeminiRow p_row;
        if (isLittleEndian) {
            p_row.timestamp = swap_uint64(row.ts);
            p_row.bid = swap_double(row.bid);
            p_row.bid_qty = swap_double(row.bid_qty);
            p_row.ask = swap_double(row.ask);
            p_row.ask_qty = swap_double(row.ask_qty);
        } else {
            p_row.timestamp = row.ts;
            p_row.bid = row.bid;
            p_row.bid_qty = row.bid_qty;
            p_row.ask = row.ask;
            p_row.ask_qty = row.ask_qty;
        }
        raw_data_.push_back(p_row);
    }
}

void GeminiDataset::precompute_features() {
    size_t n = raw_data_.size();
    features_ = torch::zeros({(long)n, 7}, torch::kFloat32);

    auto features_acc = features_.accessor<float, 2>();

    for (size_t i = 0; i < n; ++i) {
        float bid = (float)raw_data_[i].bid;
        float bid_qty = (float)raw_data_[i].bid_qty;
        float ask = (float)raw_data_[i].ask;
        float ask_qty = (float)raw_data_[i].ask_qty;
        
        float mid_price = (bid + ask) / 2.0f;
        float spread = ask - bid;
        
        float total_qty = bid_qty + ask_qty;
        if (total_qty == 0) total_qty = 1.0f;
        float imbalance = (bid_qty - ask_qty) / total_qty;

        features_acc[i][0] = bid;
        features_acc[i][1] = bid_qty;
        features_acc[i][2] = ask;
        features_acc[i][3] = ask_qty;
        features_acc[i][4] = mid_price;
        features_acc[i][5] = spread;
        features_acc[i][6] = imbalance;
    }
}

void GeminiDataset::compute_covariates() {
    size_t n = raw_data_.size();
    covariates_ = torch::zeros({(long)n, 4}, torch::kFloat32);

    auto covariates_acc = covariates_.accessor<float, 2>();

    for (size_t i = 0; i < n; ++i) {
        double timestamps_sec = raw_data_[i].timestamp / 1000.0;
        
        // C++ equivalent of datetime extraction
        // Assuming UTC
        time_t raw_time = (time_t)timestamps_sec;
        struct tm * ptm = gmtime(&raw_time);

        float cov_sec = (float)ptm->tm_sec;
        float cov_min = (float)ptm->tm_min;
        float cov_hour = (float)ptm->tm_hour;
        float cov_day = (float)ptm->tm_wday; // 0-6, Sunday is 0

        covariates_acc[i][0] = cov_sec / 60.0f - 0.5f;
        covariates_acc[i][1] = cov_min / 60.0f - 0.5f;
        covariates_acc[i][2] = cov_hour / 24.0f - 0.5f;
        covariates_acc[i][3] = cov_day / 7.0f - 0.5f;
    }
}

void GeminiDataset::normalize(size_t train_size) {
    if (train_size > features_.size(0)) {
        train_size = features_.size(0);
    }
    
    // Normalize based on training data
    auto train_slice = features_.slice(0, 0, train_size);
    auto mean = train_slice.mean(0);
    auto std = train_slice.std(0);

    // Prevent div by zero
    std = torch::where(std == 0, torch::ones_like(std), std);

    // Apply to whole dataset
    features_ = (features_ - mean) / std;
    
    std::cout << "Computed Normalization Stats (Train only):" << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Std: " << std << std::endl;
}

torch::data::Example<torch::Tensor, torch::Tensor> GeminiDataset::get(size_t index) {
    // Window: [index, index + seq_len]
    auto x = features_.slice(0, index, index + seq_len_);
    auto t = covariates_.slice(0, index, index + seq_len_);
    return {x, t}; // Features, Covariates
}

torch::optional<size_t> GeminiDataset::size() const {
    if (raw_data_.size() < (size_t)seq_len_) return 0;
    return raw_data_.size() - seq_len_ + 1;
}
