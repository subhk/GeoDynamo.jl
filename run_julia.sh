#!/bin/bash
# Helper script to run Julia with the correct depot path
# This avoids OpenSSL_jll and other package depot issues

JULIA_DEPOT_PATH=".julia_depot:$HOME/.julia" julia --project=. "$@"
