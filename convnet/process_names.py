#!/usr/bin/env python3
import os
import sys

if len(sys.argv) < 3:
  print("Usage: <exec> path prefix")
  exit()

path = sys.argv[1]
prefix = sys.argv[2]

if not os.path.exists(path):
  print("Expected path")
  exit()

files = []
for (dirpath, dirnames, filenames) in os.walk(path):
  files.extend(filenames)
  break
for idx,fn in enumerate(files):
  filename, file_extension = os.path.splitext(fn)
  name = "0000" + str(idx)
  src = os.path.join(path,fn)
  target = os.path.join(
    path,"{}{}{}".format(prefix,name[-4:],file_extension)
  )
  os.rename(src,target)
