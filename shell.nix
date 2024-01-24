with import <nixpkgs> { };

let
  pythonPackages = python310Packages;
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
in pkgs.mkShell rec {
  name = "impurePythonEnv";
  venvDir = "./.venv";
  buildInputs = [
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

    # In this particular example, in order to compile any binary extensions they may
    # require, the Python modules listed in the hypothetical requirements.txt need
    # the following packages to be installed locally:
    taglib
    openssl
    libxml2
    libxslt
    libzip
    zlib
    glibc
    glib
    zlib
    libGL
    libuuid
    stdenv.cc.cc.lib
    gcc-unwrapped
  ];

  # Run this command, only after creating the virtual environment
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r shell-requirements.txt
    pip install --upgrade -q git+https://github.com/keras-team/keras-cv
  '';

  # Now we can execute any commands within the virtual environment.
  # This is optional and can be left out to run pip manually.
  postShellHook = ''
    # allow pip to install wheels
    unset SOURCE_DATE_EPOCH
  '';

}
