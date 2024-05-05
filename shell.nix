with import <nixpkgs> {};

mkShell {
  name = "dev-shell";
  buildInputs = [ 
    ( opencv4.override { enableGtk2 = true; } )
    llvmPackages.openmp
    pkg-config 
    stdenv
    cmake 
    boost
  ];
}
