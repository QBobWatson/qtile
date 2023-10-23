{
  description = "Hacking on qtile.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-2305.url = "github:NixOS/nixpkgs/nixos-23.05";
  };

  outputs = { self, nixpkgs, nixpkgs-2305 }:
    let
      pkgs2305 = nixpkgs-2305.legacyPackages.x86_64-linux;

      pkgs = import nixpkgs {
        localSystem = pkgs2305.system;
        # need 23.05 version of mesa
        overlays = [ (self: super: { mesa = pkgs2305.mesa; }) ];
      };

      lib = pkgs.lib;
    in {
      devShell.x86_64-linux = pkgs.mkShell {
        buildInputs = with pkgs; with python3Packages; [
          pkg-config
          setuptools-scm
          setuptools
          (cairocffi.override { withXcffib = true; })
          dbus-next
          dbus-python
          iwlib
          mpd2
          psutil
          pulsectl-asyncio
          pygobject3
          python-dateutil
          pywayland
          pywlroots
          pyxdg
          xcffib
          xkbcommon
          libinput
          libxkbcommon
          wayland
          wlroots
          xorg.xcbutilwm
          glib
          pango
          xorg.xcbutilcursor
          pixman
          libdrm
        ];

        shellHook = with lib; with pkgs; ''
          export SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0
          export LD_LIBRARY_PATH=${
            makeLibraryPath [
              glib pango xorg.xcbutilcursor
            ]
          }
          export CPATH=${getDev pixman}/include/pixman-1:${getDev libdrm}/include/libdrm
        '';
      };
    };
}
