{
  description = "3D Gaussian Splatting with BVH optimization in Slang";

  inputs =
    {
      nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
      nixpkgs-cuda.url = "github:nixos/nixpkgs/nixos-24.11";
      flake-utils.url = "github:numtide/flake-utils";
      slangpy-src = {
        url = "git+file:./external/slangpy?submodules=1";
        flake = false;
      };
    };

  outputs = { self, nixpkgs, nixpkgs-cuda, flake-utils, ... } @ inputs:
    with flake-utils.lib;
    eachSystem [
      system.x86_64-linux
      system.x86_64-darwin
      system.aarch64-darwin
    ]
      (system:
        let
          inherit (nixpkgs) lib;
          pkgs = import nixpkgs {
            inherit system;
          };
          pkgs-cuda = import nixpkgs-cuda
            {
              inherit system;
              config.allowUnfree = true;
            };
          slangpy-config = import "${inputs.slangpy-src}/config.nix" {
            inherit pkgs pkgs-cuda lib;
          };
          slangpy-basePkgs = slangpy-config.basePkgs;
          slangpy-linuxPkgs = slangpy-config.linuxPkgs;
          slangpy-ldLibs = slangpy-config.ldLibs;
          basePkgs = with pkgs; [
            # Python environment.
            python3
            uv
          ];
        in
        {
          devShells.default = pkgs.mkShell
            {
              buildInputs = basePkgs ++
                slangpy-basePkgs ++
                (lib.optional pkgs.stdenv.isLinux slangpy-linuxPkgs);
              shellHook = ''
                # Create the virtual environment if it doesn't exist
                if [ -d .venv ]; then
                  # Activate the virtual environment
                  source .venv/bin/activate
                  # Add .venv/bin to PATH
                  export PATH=$PWD/.venv/bin:$PATH
                else
                  echo "Environment not initialized."
                fi
              '';
              LD_LIBRARY_PATH = lib.makeLibraryPath (
                slangpy-basePkgs ++
                (lib.optional pkgs.stdenv.isLinux slangpy-linuxPkgs) ++
                slangpy-ldLibs ++ [
                  pkgs.imath
                ]
              );
              CUDA_PATH = lib.optionalString pkgs.stdenv.isLinux pkgs-cuda.cudatoolkit;
            };
        }
      );
}
