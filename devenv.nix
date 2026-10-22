{ config, pkgs, ... }:

{
  languages.c.enable = true;

  languages.python = {
    enable = true;
    package = pkgs.python311;
    uv.enable = true;
  };

  packages = with pkgs; [
    actionlint
    coreutils
    curl
    git
    gnumake
    unzip
  ];

  # Use Nix's Python for uv projects, including the initial scaffold.
  env.UV_PYTHON = "${config.languages.python.package}/bin/python3";
  env.UV_PYTHON_DOWNLOADS = "never";
}
