let
  pkgs = import ./nix { config.allowUnfree = true; };
  python-packages = python-packages:
    with python-packages; [
      pip
      setuptools
      tkinter
      #scipy
      #pillow
      #matplotlib
      #pyyaml
      #dlib
      #(tensorflow-bin.override { cudaSupport = true; })
      #(tensorflow.override { cudaSupport = true; })
      pytorchWithCuda
      pygame
    ];
  python-with-packages = (pkgs.python35.withPackages python-packages);
  # .override (args: { ignoreCollissions = true; });
in pkgs.mkShell {
  buildInputs = with pkgs; [
    python-with-packages
    cmake
    python3Packages.black
    python3Packages.python-language-server
  ];
  PIP_PREFIX = toString ./_build/pip_packages;
  shellHook = with pkgs; ''
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${glib.out}/lib:${xlibs.libSM.out}/lib:${xlibs.libICE.out}/lib:${xlibs.libXext.out}/lib:${stdenv.cc.cc.lib}/lib:${cudatoolkit_10_0}/lib:${cudnn_cudatoolkit_10_0}/lib:${cudatoolkit_10_0.lib}/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${libpng_apng.out}/lib:${libjpeg_original.out}/lib:${libtiff.out}/lib:${xlibs.libXrender.out}/lib:${xlibs.libX11.out}/lib
    export PYTHONPATH="${
      toString ./_build/pip_packages/lib/python3.5/site-packages
    }:${toString ./_build/python3.5/site-packages}:${
      toString ./PythonAPI
    }:$PYTHONPATH"
    export PATH="${toString ./_build/pip_packages/bin}:$PATH"
    unset TMP
    unset TMPDIR
    unset TEMP
    unset TEMPDIR
    unset SOURCE_DATE_EPOCH
  '';
}
