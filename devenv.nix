{ config, pkgs, ... }:

{
  languages.c.enable = true;

  # The pinned ds4 source uses macOS 15 Metal APIs.
  apple.sdk =
    if pkgs.stdenv.isDarwin
    then pkgs.apple-sdk_15
    else null;

  languages.python = {
    enable = true;
    package = pkgs.python3;
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

  enterShell = "export MACOSX_DEPLOYMENT_TARGET=15.0";

  # Use Nix's Python for uv projects, including the initial scaffold.
  env.UV_PYTHON = "${config.languages.python.package}/bin/python3";
  env.UV_PYTHON_DOWNLOADS = "never";
}
