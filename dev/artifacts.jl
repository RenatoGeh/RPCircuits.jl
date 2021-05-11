using Pkg.Artifacts

function create_artifacts()
  artifact_toml = normpath("$(@__DIR__)/../Artifacts.toml")
  println("Artifacts.toml path: ", artifact_toml)

  zoo_hash = artifact_hash("zoo", artifact_toml)
  println("Zoo hash: ", zoo_hash)

  if zoo_hash == nothing || !artifact_exists(zoo_hash)
    println("  Zoo not found. Creating...")
    zoo_hash = create_artifact() do artifact_dir
      url_base = "https://www.ime.usp.br/~renatolg/rpc/"
      println("Downloading tarball from ", url_base, "...")
      download("$(url_base)/zoo.tar.gz", joinpath(artifact_dir, "zoo.tar.gz"))
    end
    println("Binding tarball...")
    bind_artifact!(artifact_toml, "zoo", zoo_hash)
  end
end

function main()
  println("Creating necessary artifacts...")
  create_artifacts()
end

main()
