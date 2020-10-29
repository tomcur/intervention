let
  pkgs = import ./nix { config.allowUnfree = true; };
  python-packages = python-packages:
    with python-packages; [
      pip
      setuptools
      tkinter
      pygame
    ];
  python-with-packages = (pkgs.python37.withPackages python-packages);
  libjpeg_original_8d_pkg = { stdenv, fetchurl, static ? false }:

    with stdenv.lib;

    stdenv.mkDerivation {
      name = "libjpeg-8d";

      src = fetchurl {
        url = "http://www.ijg.org/files/jpegsrc.v8d.tar.gz";
        sha256 = "1cz0dy05mgxqdgjf52p54yxpyy95rgl30cnazdrfmw7hfca9n0h0";
      };

      configureFlags = optional static "--enable-static --disable-shared";

      outputs = [ "bin" "dev" "out" "man" ];

      meta = {
        homepage = "http://www.ijg.org/";
        description = "A library that implements the JPEG image file format";
        license = stdenv.lib.licenses.free;
        platforms = stdenv.lib.platforms.unix;
      };
    };
  libjpeg_original_8d = pkgs.callPackage libjpeg_original_8d_pkg { };
  libproj_pkg = { stdenv, fetchurl }:
    stdenv.mkDerivation {
      name = "proj-4.9.1";

      src = fetchurl {
        url = "http://download.osgeo.org/proj/proj-4.9.1.tar.gz";
        sha256 = "06f36s7yi6yky92g235kj9wkcckm04qgzxnj0fla3icb7y7ki87w";
      };

      meta = with stdenv.lib; {
        description = "Cartographic Projections Library";
        homepage = "http://trac.osgeo.org/proj/";
        license = licenses.mit;
        platforms = platforms.linux;
      };
    };
  libproj = pkgs.callPackage libproj_pkg { };
in pkgs.mkShell {
  venvDir = "./.venv";
  buildInputs = with pkgs; [
    python-with-packages
    cmake
    python37Packages.black
    python37Packages.python-language-server
    python37Packages.venvShellHook
  ];
  MYPYPATH = toString ./stubs;
  LD_LIBRARY_PATH = with pkgs;
    "/run/opengl-driver/lib:${glib.out}/lib:${xlibs.libSM.out}/lib:${xlibs.libICE.out}/lib"
    + ":${xlibs.libXext.out}/lib:${stdenv.cc.cc.lib}/lib:${libpng_apng.out}/lib:${libjpeg_original_8d.out}/lib"
    + ":${libtiff.out}/lib:${xlibs.libXrender.out}/lib:${xlibs.libX11.out}/lib"
    # For Carla RSS integration:
    + ":${tbb.out}/lib:${libproj}/lib";
}
