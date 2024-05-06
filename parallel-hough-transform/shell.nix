with import <nixpkgs> {};

mkShell {
  name = "dev-shell";
  buildInputs = [ 
    ( opencv4.override { enableGtk2 = true; } )
    llvmPackages.openmp
    mpi
    pkg-config 
    stdenv
    cmake 
    boost
  ];
}
