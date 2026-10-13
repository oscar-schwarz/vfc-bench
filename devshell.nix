{ pkgs }:
let
  python = pkgs.python3.withPackages (ps: [
    # Runtime deps, mirroring [project.dependencies]
    # Dev deps
    ps.pytest
  ]);
in
pkgs.mkShell {
  packages = [
    python
    pkgs.ruff
  ];

  env = { };

  shellHook = ''
    # src layout: importable without an install step
    export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
  '';
}
