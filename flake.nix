{
  description = "3D Gaussian Splatting with BVH optimization in Slang";

  inputs =
    {
      nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
      flake-utils.url = "github:numtide/flake-utils";
    };

  outputs = { self, nixpkgs, flake-utils }:
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
          ldPath = lib.optionalString pkgs.stdenv.isLinux (
            lib.makeLibraryPath
              (
                pkgs.pythonManylinuxPackages.manylinux1 ++
                [
                  "/run/opengl-driver"
                  pkgs.vulkan-loader
                ]
              )
          );
        in
        {
          devShells.default = pkgs.mkShell
            {
              buildInputs = with pkgs; [
                # Python environment.
                python3
                uv
              ] ++ (lib.optional pkgs.stdenv.isLinux [
                xorg.libX11
              ]);
              shellHook = ''
                # Create the virtual environment if it doesn't exist
                if [ -d .venv ]; then
                  # Activate the virtual environment
                  source .venv/bin/activate
                  # Add .venv/bin to PATH
                  export PATH=$PWD/.venv/bin:$PATH
                  export LD_LIBRARY_PATH=${ldPath}:$LD_LIBRARY_PATH
                else
                  echo "Environment not initialized."
                fi
              '';
            };
        }
      );
}
