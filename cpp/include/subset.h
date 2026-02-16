#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <vector>
#include <algorithm>

namespace torch {
namespace data {
namespace datasets {

/// A dataset that is a subset of another dataset.
/// It wraps the original dataset and a list of indices, and only yields
/// the elements from the original dataset at those indices.
template <typename DatasetType>
class Subset : public Dataset<Subset<DatasetType>, typename DatasetType::ExampleType> {
 public:
  using ExampleType = typename DatasetType::ExampleType;
  using BatchType = typename DatasetType::BatchType;
  using BatchRequestType = typename DatasetType::BatchRequestType;

  /// Constructs a `Subset` from a dataset and a list of indices.
  /// The indices are copied.
  Subset(DatasetType dataset, std::vector<size_t> indices)
      : dataset_(std::move(dataset)), indices_(std::move(indices)) {}

  /// Returns the example at the given index (which is an index into the *subset*,
  /// not the original dataset).
  ExampleType get(size_t index) override {
    return dataset_.get(indices_.at(index));
  }
  
  /// Returns the size of the subset.
  std::optional<size_t> size() const override {
    return indices_.size();
  }



    // We need to expose the underlying dataset type to properly inherit/use traits if needed,
    // but for now let's just implement the basics required by the DataLoader.

 const DatasetType& dataset() const { return dataset_; }
 const std::vector<size_t>& indices() const { return indices_; }

 private:
  DatasetType dataset_;
  std::vector<size_t> indices_;
};

} // namespace datasets
} // namespace data
} // namespace torch
