#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <iomanip>
#include <ctime>
#include <sstream>
#include "subset.h"
#include "dataset.h"
#include "model.h"

// Hyperparameters
struct Args {
    std::string data_path = "data/btcusd/2026-02-16.bin";
    int epochs = 10;
    int batch_size = 32;
    double lr = 1e-4;
    std::string device = "cuda";
};

void train_epoch(
    Pyraformer model,
    torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<GeminiDataset, torch::data::transforms::Stack<torch::data::Example<>>>, torch::data::samplers::RandomSampler>& loader,
    std::shared_ptr<torch::optim::Adam> optimizer,
    torch::nn::HuberLoss loss_fn,
    int epoch,
    torch::Device device
) {
    model->train();
    double total_loss = 0;
    int batches = 0;

    // Iterate through the data loader
    for (auto& batch : loader) {
        auto data = batch.data.to(device);
        auto covariates = batch.target.to(device); // Check: GeminiDataset::get returns {features, covariates}

        // data: [batch, seq_len, 7]
        // covariates: [batch, seq_len, 4]

        int64_t input_size = model->input_size();

        auto x_enc = data.slice(1, 0, input_size);
        auto t_enc = covariates.slice(1, 0, input_size);
        
        auto target = data.slice(1, input_size, torch::indexing::None);
        
        // x_dec preparation
        auto x_dec = torch::zeros_like(target).to(device);
        int64_t label_len = input_size / 2;
        // x_dec[:, :label_len, :] = x_enc[:, -label_len:, :]
        x_dec.index({torch::indexing::Slice(), torch::indexing::Slice(0, label_len), torch::indexing::Slice()}) = 
            x_enc.index({torch::indexing::Slice(), torch::indexing::Slice(-label_len, torch::indexing::None), torch::indexing::Slice()});
            
        auto t_dec = covariates.slice(1, input_size, torch::indexing::None);

        optimizer->zero_grad();

        auto pred = model->forward(x_enc, t_enc, x_dec, t_dec, false);

        auto loss = loss_fn->forward(pred, target);
        
        loss.backward();
        optimizer->step();

        total_loss += loss.item().toDouble();
        batches++;

        if (batches % 10 == 0) {
            std::cout << "\rEpoch " << epoch << " [Train] Batch " << batches << " Loss: " << std::fixed << std::setprecision(6) << loss.item().toDouble() << std::flush;
        }
    }
    std::cout << std::endl;
    std::cout << "Epoch " << epoch << " Train Loss: " << total_loss / batches << std::endl;
}

void eval_epoch(
    Pyraformer model,
    torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::Subset<GeminiDataset>, torch::data::transforms::Stack<torch::data::Example<>>>, torch::data::samplers::SequentialSampler>& loader,
    torch::nn::HuberLoss loss_fn,
    int epoch,
    torch::Device device
) {
    model->eval();
    double total_loss = 0;
    int batches = 0;

    {
        torch::NoGradGuard no_grad;
        for (auto& batch : loader) {
            auto data = batch.data.to(device);
            auto covariates = batch.target.to(device);

            int64_t input_size = model->input_size();

            auto x_enc = data.slice(1, 0, input_size);
            auto t_enc = covariates.slice(1, 0, input_size);
            
            auto target = data.slice(1, input_size, torch::indexing::None);
            
            auto x_dec = torch::zeros_like(target).to(device);
            int64_t label_len = input_size / 2;
            x_dec.index({torch::indexing::Slice(), torch::indexing::Slice(0, label_len), torch::indexing::Slice()}) = 
                x_enc.index({torch::indexing::Slice(), torch::indexing::Slice(-label_len, torch::indexing::None), torch::indexing::Slice()});
                
            auto t_dec = covariates.slice(1, input_size, torch::indexing::None);

            auto pred = model->forward(x_enc, t_enc, x_dec, t_dec, false);
            auto loss = loss_fn->forward(pred, target);

            total_loss += loss.item().toDouble();
            batches++;
        }
    }
    std::cout << "Epoch " << epoch << " Val Loss: " << total_loss / batches << std::endl;
}

int main(int argc, char* argv[]) {
    Args args;
    if (argc > 1) args.data_path = argv[1];
    if (argc > 2) args.epochs = std::stoi(argv[2]);
    if (argc > 3) args.batch_size = std::stoi(argv[3]);

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available() && args.device == "cuda") {
        std::cout << "Using CUDA" << std::endl;
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "Using CPU" << std::endl;
    }

    std::cout << "Loading data from " << args.data_path << "..." << std::endl;
    auto dataset = GeminiDataset(args.data_path, 2048, 1024);
    
    auto total_len = dataset.size().value();
    auto train_size = static_cast<size_t>(0.8 * total_len);
    auto val_size = total_len - train_size;

    std::cout << "Total samples: " << total_len << ". Train: " << train_size << ", Val: " << val_size << std::endl;

    dataset.normalize(train_size);

    // Split
    // Subset requires a vector of indices
    std::vector<size_t> train_indices(train_size);
    std::iota(train_indices.begin(), train_indices.end(), 0);
    
    std::vector<size_t> val_indices(val_size);
    std::iota(val_indices.begin(), val_indices.end(), train_size);

    auto train_dataset = torch::data::datasets::Subset<GeminiDataset>(dataset, train_indices);
    auto val_dataset = torch::data::datasets::Subset<GeminiDataset>(dataset, val_indices);

    auto train_loader = torch::data::make_data_loader(
        train_dataset.map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(args.batch_size).workers(8)
    );

    auto val_loader = torch::data::make_data_loader(
        val_dataset.map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(args.batch_size).workers(8)
    );

    // Model
    ModelOptions opt;
    opt.input_size = 2048;
    opt.predict_size = 1024;
    opt.d_model = 512;
    opt.d_inner_hid = 512;
    opt.d_k = 128;
    opt.d_v = 128;
    opt.n_head = 6;
    opt.n_layer = 6;
    opt.window_size = {4, 4, 4};
    opt.dropout = 0.05;
    opt.enc_in = 7;
    opt.device = device;
    opt.inner_size = 3; // From python default
    opt.d_bottleneck = 128; // From python default

    Pyraformer model(opt);
    model->to(device);

    auto optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(args.lr));
    auto loss_fn = torch::nn::HuberLoss(torch::nn::HuberLossOptions().delta(1.0));

    std::cout << "Starting training..." << std::endl;

    for (int epoch = 1; epoch <= args.epochs; ++epoch) {
        // Need to pass the specific loader type or use auto/template
        // Passing pointers to loader is tricky with make_data_loader return type which is unique_ptr
        // We'll just define logic here or use templates.
        // For simplicity, let's just inline logic or fix the function signature.
        // The signature in `train_epoch` uses specific types which might mismatch `make_data_loader` return types.
        // `make_data_loader` returns `std::unique_ptr<StatelessDataLoader<...>>`.
        
        // Let's call data loader directly here to avoid type mess
        model->train();
        double train_loss_sum = 0;
        int train_batches = 0;
        
        for (auto& batch : *train_loader) {
             auto data = batch.data.to(device);
             auto covariates = batch.target.to(device);

             int64_t input_size = model->input_size();

             auto x_enc = data.slice(1, 0, input_size);
             auto t_enc = covariates.slice(1, 0, input_size);
             auto target = data.slice(1, input_size, torch::indexing::None);
             
             auto x_dec = torch::zeros_like(target).to(device);
             int64_t label_len = input_size / 2;
             x_dec.index({torch::indexing::Slice(), torch::indexing::Slice(0, label_len), torch::indexing::Slice()}) = 
                 x_enc.index({torch::indexing::Slice(), torch::indexing::Slice(-label_len, torch::indexing::None), torch::indexing::Slice()});
             
             auto t_dec = covariates.slice(1, input_size, torch::indexing::None);

             optimizer->zero_grad();
             auto pred = model->forward(x_enc, t_enc, x_dec, t_dec, false);
             auto loss = loss_fn->forward(pred, target);
             loss.backward();
             optimizer->step();

             train_loss_sum += loss.item().toDouble();
             train_batches++;
             
             if (train_batches % 10 == 0) std::cout << "." << std::flush;
        }
        std::cout << " Train Loss: " << train_loss_sum / train_batches << "\t";

        // Validation
        model->eval();
        double val_loss_sum = 0;
        int val_batches = 0;
        {
            torch::NoGradGuard no_grad;
            for (auto& batch : *val_loader) {
                 auto data = batch.data.to(device);
                 auto covariates = batch.target.to(device);

                 int64_t input_size = model->input_size();

                 auto x_enc = data.slice(1, 0, input_size);
                 auto t_enc = covariates.slice(1, 0, input_size);
                 auto target = data.slice(1, input_size, torch::indexing::None);
                 
                 auto x_dec = torch::zeros_like(target).to(device);
                 int64_t label_len = input_size / 2;
                 x_dec.index({torch::indexing::Slice(), torch::indexing::Slice(0, label_len), torch::indexing::Slice()}) = 
                     x_enc.index({torch::indexing::Slice(), torch::indexing::Slice(-label_len, torch::indexing::None), torch::indexing::Slice()});
                 
                 auto t_dec = covariates.slice(1, input_size, torch::indexing::None);

                 auto pred = model->forward(x_enc, t_enc, x_dec, t_dec, false);
                 auto loss = loss_fn->forward(pred, target);
                 
                 val_loss_sum += loss.item().toDouble();
                 val_batches++;
            }
        }
        std::cout << "Val Loss: " << val_loss_sum / val_batches << std::endl;
    }

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << "pyraformer_checkpoint_" << std::put_time(&tm, "%Y-%m-%d") << ".pt";
    torch::save(model, oss.str());
    std::cout << "Model saved to " << oss.str() << std::endl;

    return 0;
}
