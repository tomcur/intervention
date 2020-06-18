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
  python-with-packages = (pkgs.python37.withPackages python-packages);
in pkgs.mkShell {
  venvDir = "./.venv";
  buildInputs = with pkgs; [
    python-with-packages
    cmake
    python37Packages.black
    python37Packages.python-language-server
    python37Packages.venvShellHook
  ];
  LD_LIBRARY_PATH = with pkgs;
    "${glib.out}/lib:${xlibs.libSM.out}/lib:${xlibs.libICE.out}/lib:${xlibs.libXext.out}/lib:${stdenv.cc.cc.lib}/lib:${libpng_apng.out}/lib:${libjpeg_original.out}/lib:${libtiff.out}/lib:${xlibs.libXrender.out}/lib:${xlibs.libX11.out}/lib";
}
