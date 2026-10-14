{ pkgs }:
let
  python = pkgs.python3.withPackages (ps: [
    # Runtime deps, mirroring [project.dependencies]
    # Dev deps
    ps.pytest
    ps.python-dotenv # examples extra
    ps.openai # examples extra
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
