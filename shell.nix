{ pkgs ? import <nixpkgs> {
  config = {
    allowUnfree = true;
    cudaSupport = false;
  };
} }:
  pkgs.mkShell {
    # nativeBuildInputs is usually what you want -- tools you need to run
    nativeBuildInputs = with pkgs.buildPackages; [
      python312
      python312Packages.torch-bin
      python312Packages.fairscale
      python312Packages.fire
      python312Packages.blobfile
      python312Packages.pip
      cudaPackages.cudatoolkit
      python312Packages.llamaindex-py-client
      python312Packages.llama-index-cli
      python312Packages.llama-cpp-python
    ];

    shellHook = ''
      echo "You are now using a NIX environment"
      export CUDA_PATH=${pkgs.cudatoolkit}
      export TMPDIR=/tmp
        export VENV_DIR=$(mktemp -d)
        python -m venv $VENV_DIR
        source $VENV_DIR/bin/activate
        echo "Virtual environment is ready and activated in $VENV_DIR."
        pip install llama-stack 
    '';
}