{
  inputs = {
    nixpkgs = { url = "github:nixos/nixpkgs/nixos-23.11"; };
    flake-utils = { url = "github:numtide/flake-utils"; };
  };
  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        pythonPackages = pkgs.python310Packages;
        python = let
          packageOverrides = self: super: {
            opencv4 = super.opencv4.override {
              enableGtk2 = true;
              gtk2 = pkgs.gtk2;
              #enableFfmpeg = true; #here is how to add ffmpeg and other compilation flags
              #ffmpeg_3 = pkgs.ffmpeg-full;
            };
          };
        in pkgs.python310.override {
          inherit packageOverrides;
          self = python;
        };
      in {
        devShell = pkgs.mkShell {
          name = "impurePythonEnv";
          venvDir = "./.venv";
          nativeBuildInputs = [
            (python.withPackages (ps: with ps; [ opencv4 ]))
            # A Python interpreter including the 'venv' module is required to bootstrap
            # the environment.
            pythonPackages.python
            pythonPackages.black

            # This executes some shell code to initialize a venv in $venvDir before
            # dropping into the shell
            pythonPackages.venvShellHook

            # Those are dependencies that we would like to use from nixpkgs, which will
            # add them to PYTHONPATH and thus make them accessible from within the venv.
            # pythonPackages.numpy
            # pythonPackages.opencv4
            # pythonPackages.requests
            pythonPackages.matplotlib
            pythonPackages.scikit-learn
            pythonPackages.pandas
            pythonPackages.jupyter

            # In this particular example, in order to compile any binary extensions they may
            # require, the Python modules listed in the hypothetical requirements.txt need
            # the following packages to be installed locally:
            pkgs.taglib
            pkgs.openssl
            pkgs.libxml2
            pkgs.libxslt
            pkgs.libzip
            pkgs.zlib
            pkgs.glibc
            pkgs.glib
            pkgs.zlib
            pkgs.libGL
            pkgs.libuuid
            pkgs.stdenv.cc.cc.lib
            pkgs.gcc-unwrapped
            pkgs.gcc-unwrapped.lib
            pkgs.libgccjit.out
          ];

          # Run this command, only after creating the virtual environment
          postVenvCreation = ''
            unset SOURCE_DATE_EPOCH
            pip install -r requirements.txt
            pip install --upgrade -q git+https://github.com/keras-team/keras-cv
          '';

          # Now we can execute any commands within the virtual environment.
          # This is optional and can be left out to run pip manually.
          postShellHook = ''
            # allow pip to install wheels
            unset SOURCE_DATE_EPOCH
          '';
        };
      });
}
