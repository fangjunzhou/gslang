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
          basePkgs = with pkgs; [
            # Python environment.
            python3
            uv
            # Build system.
            cmake
            ninja
            # Slangpy dependencies.
            libjpeg
            libpng
            openexr_3
            asmjit
          ];
          linuxPkgs = with pkgs; [
            xorg.libX11
          ];
        in
        {
          devShells.default = pkgs.mkShell
            {
              buildInputs = basePkgs ++ (lib.optional pkgs.stdenv.isLinux linuxPkgs);
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

              LD_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isLinux (
                lib.makeLibraryPath
                  (
                    pkgs.pythonManylinuxPackages.manylinux1 ++
                    [
                      "/run/opengl-driver"
                      pkgs.vulkan-loader
                    ]
                  )
              );
            };
        }
      );
}
