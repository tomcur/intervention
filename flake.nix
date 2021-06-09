{
  description = "Learning to drive from interventions";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.flake-compat = {
    url = "github:edolstra/flake-compat";
    flake = false;
  };
  inputs.libjpeg = {
    url = "http://www.ijg.org/files/jpegsrc.v8d.tar.gz";
    flake = false;
  };
  inputs.libproj = {
    url = "http://download.osgeo.org/proj/proj-4.9.1.tar.gz";
    flake = false;
  };
  outputs = { self, nixpkgs, flake-utils, libjpeg, libproj, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
      in
      rec {
        packages.libjpeg = pkgs.callPackage
          ({ lib, stdenv, static ? false }:
            with lib;
            stdenv.mkDerivation {
              name = "libjpeg-8d";
              src = libjpeg;
              configureFlags = optional static "--enable-static --disable-shared";
              outputs = [ "bin" "dev" "out" "man" ];
              meta = {
                homepage = "http://www.ijg.org/";
                description =
                  "A library that implements the JPEG image file format";
                license = licenses.free;
                platforms = platforms.unix;
              };
            })
          { };
        packages.libproj = pkgs.stdenv.mkDerivation {
          name = "proj-4.9.1";
          src = libproj;
          meta = with pkgs.lib; {
            description = "Cartographic Projections Library";
            homepage = "http://trac.osgeo.org/proj/";
            license = licenses.mit;
            platforms = platforms.linux;
          };
        };
        packages.pythonWithPackages =
          let
            python-packages = python-packages:
              with python-packages; [
                pip
                setuptools
                tkinter
                pygame
              ];
          in
          (pkgs.python37.withPackages python-packages);
        devShell = pkgs.mkShell {
          venvDir = "./.venv";
          buildInputs = with pkgs; [
            packages.pythonWithPackages
            cmake
            python37Packages.black
            python37Packages.python-language-server
            python37Packages.jupyter
            python37Packages.venvShellHook
          ];
          MYPYPATH = toString ./stubs;
          LD_LIBRARY_PATH =
            let
              libraries = [
                # Libraries
                pkgs.stdenv.cc.cc.lib
                pkgs.glib.out
                pkgs.xlibs.libSM.out
                pkgs.xlibs.libICE.out
                pkgs.xlibs.libXext.out
                pkgs.libpng_apng.out
                packages.libjpeg.out
                pkgs.libtiff.out
                pkgs.xlibs.libXrender.out
                pkgs.xlibs.libX11.out
                # For Carla RSS integration:
                pkgs.tbb.out
                packages.libproj
              ];
            in
            "/run/opengl-driver/lib:"
            + (pkgs.lib.concatMapStringsSep ":" (pkg: "${pkg}/lib")
              libraries);
          postShellHook = ''
            PYTHONPATH=$PWD/$venvDir/${packages.pythonWithPackages.sitePackages}:$PYTHONPATH
          '';
        };
      });
}
