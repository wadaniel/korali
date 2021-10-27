#!/usr/bin/env bash
set -e
podman rmi -f cselab/korali:latest
podman rmi -f cselab/korali:3.0.0
