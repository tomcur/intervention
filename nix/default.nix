{ sources ? import ./sources.nix, system ? builtins.currentSystem, config ? { }
}:
let
  srcs = self: super: { inherit sources; };
  overlays = [ srcs ];
in import sources.nixpkgs { inherit overlays config system; }
