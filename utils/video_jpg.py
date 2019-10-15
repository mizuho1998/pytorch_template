"""
python video_jpg.py <path/to/video/directory> <path/to/jpg/directory>
"""


from __future__ import print_function, division
import os
import sys
import subprocess
from multiprocessing import Pool
import multiprocessing as multi

def file_process(dir_path, dst_dir_path, file_name):
  file_path = os.path.join(dir_path, file_name)
  # print(file_path)
  if not os.path.exists(file_path):
    return
  
  if '.mp4' not in file_name:
    return 

  file_base_name, _ = os.path.splitext(file_name)
  dst_directory_path = os.path.join(dst_dir_path, file_base_name)
  # print(file_base_name)
 
  try:
    if os.path.exists(dst_directory_path):
      if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
        subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
        print('remove {}'.format(dst_directory_path))
        os.mkdir(dst_directory_path)
      else:
        return 
    else:
      os.mkdir(dst_directory_path)
  except:
    print(dst_directory_path)
    return 

  cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 -qscale:v 1 \"{}/image_%05d.jpg\"'.format(file_path, dst_directory_path)
  print(cmd)
  subprocess.call(cmd, shell=True)
  print('\n')


def process_wrapper(args):
  file_process(*args)


if __name__=="__main__":
  dir_path = sys.argv[1]
  dst_dir_path = sys.argv[2]

  files = os.listdir(dir_path)
  argList = [(dir_path, dst_dir_path, file_name) for _, file_name in enumerate(files)]
  # print(files)

  # p = Pool(multi.cpu_count())
  p = Pool(4)
  p.map(process_wrapper, argList)
  p.close()
