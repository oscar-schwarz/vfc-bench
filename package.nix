{ pkgs, ... }:
pkgs.python3Packages.buildPythonPackage {
  pname = "vfc-bench";
  version = "0.1.0";
  pyproject = true;

  # Keep .direnv/ and friends out of the store path.
  src = pkgs.lib.fileset.toSource {
    root = ./.;
    fileset = pkgs.lib.fileset.unions [
      ./pyproject.toml
      ./README.md
      ./src
      ./tests
    ];
  };

  build-system = [ pkgs.python3Packages.hatchling ];

  # Runtime dependencies go here, mirroring [project.dependencies].
  dependencies = [ ];

  nativeCheckInputs = [ pkgs.python3Packages.pytestCheckHook ];

  pythonImportsCheck = [ "vfc_bench" ];

  meta = {
    description = "VFC benchmarking library";
    license = pkgs.lib.licenses.mit;
  };
}
