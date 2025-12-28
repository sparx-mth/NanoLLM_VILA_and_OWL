#!/bin/bash

if [ -z "$1" ]; then
  echo "No argument provided! Map to user@192.168.131.22 (AGX1)"
  dsi_num=22
else
  dsi_num="$1"
fi

port_num="60${dsi_num}"
dsi_name="192.168.131.${dsi_num}"

echo "Map port ${port_num} to ${dsi_name}:22"

ssh -L "${port_num}:${dsi_name}:22" "user@localhost:8080"