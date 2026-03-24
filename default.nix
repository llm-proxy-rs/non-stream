let
  sources = import ./npins;
in
  {pkgs ? import sources.nixpkgs {}}: let
    lib = pkgs.lib;
    python = pkgs.python314;

    pyproject-nix = import sources."pyproject.nix" {inherit lib;};
    uv2nix = import sources.uv2nix {inherit lib pyproject-nix;};
    pyproject-build-systems = import sources.build-system-pkgs {inherit lib pyproject-nix uv2nix;};

    workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./.;};

    overlay = workspace.mkPyprojectOverlay {sourcePreference = "wheel";};

    pythonSet =
      (pkgs.callPackage pyproject-nix.build.packages {inherit python;})
    .overrideScope (
        lib.composeManyExtensions [
          pyproject-build-systems.wheel
          overlay
        ]
      );

    venv = pythonSet.mkVirtualEnv "non-stream-env" workspace.deps.default;

    entrypoint = pkgs.writeScriptBin "server" ''
      #!${pkgs.bash}/bin/bash
      exec ${venv}/bin/python ${venv}/${python.sitePackages}/server.py
    '';
  in
    pkgs.dockerTools.buildImage {
      name = "non-stream";
      tag = "latest";
      config = {
        Entrypoint = ["${entrypoint}/bin/server"];
      };
      copyToRoot = pkgs.buildEnv {
        name = "image-root";
        paths = [
          entrypoint
          pkgs.dockerTools.caCertificates
        ];
      };
    }
