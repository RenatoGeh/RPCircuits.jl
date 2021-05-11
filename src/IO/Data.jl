using DataFrames
using CSV
using LazyArtifacts

const twenty_dataset_names = [
        "accidents", "ad", "baudio", "bbc", "bnetflix", "book", "c20ng", "cr52", "cwebkb",
        "dna", "jester", "kdd", "kosarek", "msnbc", "msweb", "nltcs", "plants", "pumsb_star", "tmovie", "tretail", 
        "binarized_mnist"
];

"""
    train, valid, test = twenty_datasets(name)

Load a given dataset from the density estimation datasets. Automatically downloads the files as julia Artifacts.
See https://github.com/UCLA-StarAI/Density-Estimation-Datasets for a list of avaialble datasets.
"""
function twenty_datasets(name::String; as_df::Bool = false)::Union{Tuple{DataFrame, DataFrame, DataFrame}, Tuple{Matrix, Matrix, Matrix}}
  @assert in(name, twenty_dataset_names)
  data_dir = artifact"density_estimation_datasets"
  function load(type::String)::Union{DataFrame, Matrix}
    dataframe = CSV.read(
      data_dir * "/Density-Estimation-Datasets-1.0.1/datasets/$name/$name.$type.data",
      DataFrame;
      header = false,
      truestrings = ["1"],
      falsestrings = ["0"],
      type = Bool,
      strict = true,
    )
    # make sure the data is backed by a `BitArray`
    return as_df ? DataFrame(BitArray(Matrix{Bool}(dataframe))) : Matrix{Bool}(dataframe)
  end
  train = load("train")
  valid = load("valid")
  test = load("test")
  return train, valid, test
end
export twenty_datasets
