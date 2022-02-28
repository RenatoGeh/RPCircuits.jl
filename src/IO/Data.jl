using DataFrames
using Downloads
using CSV
using LazyArtifacts
using MLDatasets
using Images

const twenty_dataset_names = [
        "accidents", "ad", "adult", "baudio", "bbc", "bnetflix", "book", "c20ng", "cr52", "cwebkb",
        "dna", "jester", "kdd", "kosarek", "msnbc", "mushrooms", "msweb", "nltcs", "plants", "pumsb_star", "tmovie", "tretail", 
        "binarized_mnist"
];

"""
    train, valid, test = twenty_datasets(name)

Load a given dataset from the density estimation datasets. Automatically downloads the files as julia Artifacts.
See https://github.com/UCLA-StarAI/Density-Estimation-Datasets for a list of avaialble datasets.
"""
function twenty_datasets(name::String; as_df::Bool = true)::Union{Tuple{DataFrame, DataFrame, DataFrame}, Tuple{Matrix, Matrix, Matrix}}
  @assert in(name, twenty_dataset_names)
  data_dir = artifact"density_estimation_datasets"
  function load(type::String)::Union{DataFrame, Matrix}
    dataframe = CSV.read(
      data_dir * "/Density-Estimation-Datasets-1.0.1/datasets/$name/$name.$type.data",
      DataFrame;
      header = false,
      truestrings = ["1"],
      falsestrings = ["0"],
      types = Bool,
      strict = true,
    )
    # make sure the data is backed by a `BitArray`
    return as_df ? DataFrame(BitArray(Matrix{Bool}(dataframe)), :auto) : Matrix{Bool}(dataframe)
  end
  train = load("train")
  valid = load("valid")
  test = load("test")
  return train, valid, test
end
export twenty_datasets

"""
MNIST 3D `Array` to regular matrix.
"""
function mnist()::Tuple{Matrix{Float64}, Matrix{Float64}}
  X, Y = MNIST.traindata()
  d, _, n = size(X)
  m = d*d
  R = Matrix{Float64}(undef, n, m+1)
  Threads.@threads for i ∈ 1:n
    R[i,1:end-1] .= reshape(X[:,:,i], :)
    R[i,end] = Y[i]
  end
  X, Y = MNIST.testdata()
  n = size(X)[3]
  T = Matrix{Float64}(undef, n, m+1)
  Threads.@threads for i ∈ 1:n
    T[i,1:end-1] .= reshape(X[:,:,i], :)
    T[i,end] = Y[i]
  end
  return R, T
end
export mnist

"""
Returns an MNIST instance in image format.
"""
mnist_img(X::AbstractVector{Float64}) = (MNIST.convert2image(reshape(X[1:end-1], 28, 28)), X[end])
export mnist_img

const continuous_datasets_names = ["abalone", "banknote", "ca", "kinematics", "quake", "sensorless",
                                   "chemdiab", "oldfaithful", "iris", "flowsize", "triazines"]

"""
    data = continuous_datasets(name)

Load a given dataset from the continuous density estimation datasets. Automatically downloads the files as julia Artifacts.
See https://github.com/RenatoGeh/CDEBD for a list of avaialble datasets.
"""
function continuous_datasets(name::String; as_df::Bool = true, normalize::Bool = false)::Union{DataFrame, Matrix{Float64}}
  @assert in(name, continuous_datasets_names)
  url = name ∈ ["iris", "oldfaithful", "chemdiab"] ? "https://raw.githubusercontent.com/RenatoGeh/CDEBD/main/datasets/$(name).data" : "https://raw.githubusercontent.com/RenatoGeh/CDEBD/main/kfold/$(name)/$(name).data"
  D = CSV.File(Downloads.download(url)) |> DataFrame
  if normalize
    M = Matrix{Float64}(D)
    E = vec(extrema(M; dims = 1))
    Threads.@threads for x ∈ 1:size(M, 2)
      a, b = E[x]
      M[:,x] .-= a
      d = b-a
      M[:,x] ./= d
    end
    return as_df ? DataFrame(M, :auto) : M
  end
  return as_df ? D : Matrix{Float64}(D)
end
export continuous_datasets

function continuous_10fold_datasets(name::String)::Vector{Tuple{AbstractMatrix, AbstractMatrix}}
  @assert in(name, continuous_datasets_names)
  D = [Matrix{Float64}(CSV.File(Downloads.download("https://raw.githubusercontent.com/RenatoGeh/CDEBD/main/kfold/$(name)/$(name).$(i).data")) |> DataFrame) for i ∈ 1:10]
  P = Vector{Tuple{AbstractMatrix, AbstractMatrix}}(undef, 10)
  for i ∈ 1:10 P[i] = (D[i], vcat(D[begin:end .!= i]...)) end
  return P
end
export continuous_10fold_datasets

function normalize!(D::AbstractMatrix; scale::Real = 1.0)::Vector{Tuple{Float64, Float64}}
  E = vec(extrema(D; dims = 1))
  for x ∈ 1:size(D, 2)
    a, b = E[x]
    D[:,x] .-= a
    d = b-a
    D[:,x] ./= d
    D[:,x] .*= scale
  end
  return E
end
@inline function normalize(D::AbstractMatrix; scale::Real = 1.0)::Tuple{AbstractMatrix, Vector{Tuple{Float64, Float64}}}
  M = copy(D)
  return M, normalize!(M; scale)
end
export normalize, normalize!

