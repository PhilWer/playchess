'''
After having saved RGB and depth images as well as pointclouds with their timestamp in the
filename, do an offline synchronization of the collected data. This is useful to reduce the
data size as the acquisition rate of the RGBD images is considerably higher than the one of
the pointclouds, hence a lot of images does not have a cloud exactly matching them.

INPUT:
Folder tree is assumed to be something like:
~/pointing/yymmdd_pointing_to_object
	- cloud
		- cloud[timestamp].ply
		...
	- depth
		- depth[timestamp].csv
		...
	- depth_norm
		- depth_norm[timestamp].png
		...
	- rgb
		- rgb[timestamp].png
		...
	...
The format of timestamps is assumed to be s_ns.

OUTPUT:
This script does not modify the original structure, but creates a new 'synchro' folder in which
RGBD data referred to (almost) the same time instant are included in the same folder:
~/pointing/yymmdd_pointing_to_object
	- synchro
		- [timestamp]
			- cloud[timestamp].ply
			- depth[timestamp].csv
			- depth_norm[timestamp].png
			- rgb[timestamp].png
			...
		... 
			
The format of the timestamp is s_ns where with s and ns values constrained to 4 digits.
'''

import os
import shutil

DIR = '/root/tiago_public_ws/src/tiago_playchess/pcd_and_image'
DEPTH_DIR = os.path.join(DIR, 'depth')
PCD_DIR = os.path.join(DIR, 'cloud')
RGB_DIR = os.path.join(DIR, 'rgb')
DEPTH_NORM_DIR = os.path.join(DIR, 'depth_norm')

def get_timestamp(file_list, startswith, extension):
	# Extract the timestamps from the filenames
	stamps = []
	for f in file_list:
		if f.startswith(startswith):
			f = f.strip(startswith)
			f = f.strip(extension)
			timestamp = f.split('_')
			s = timestamp[0][-4:]
			ns = timestamp[1][:4] if len(timestamp[1]) == 9 else '0' + timestamp[1][:3] # the '0' is ignored in folder names... why?
			stamps.append(float(s + '.' + ns))	# don't store too many digits, the acquisition
												# last up to some minutes and the samples are
												# well spaced in time (0.1s)
	return stamps

# get the filenames in each folder
pcd_files = sorted(os.listdir(PCD_DIR))
depth_files = sorted(os.listdir(DEPTH_DIR))
rgb_files = sorted(os.listdir(RGB_DIR))
depth_norm_files = sorted(os.listdir(DEPTH_NORM_DIR))

# for each cloud, get the depth image with the closer timestamp
pcd_stamps = get_timestamp(pcd_files, 'cloud', '.ply')
rgb_stamps = get_timestamp(rgb_files, 'rgb', '.png')
depth_stamps = get_timestamp(depth_files, 'depth', '.csv')

depth_idx = []
for pcd_stamp in pcd_stamps:
	# match each cloud with the depth image having the closer-in-time stamp
	time_diffs = [abs(pcd_stamp - depth_stamp) for depth_stamp in depth_stamps]
	min_time_diff, min_time_diff_idx = min(time_diffs), time_diffs.index(min(time_diffs))
	print('Min. time diff: {time_diff:.4f}\t\tfor index: {index}'.format(	time_diff = min_time_diff, 
																			index = min_time_diff_idx))
	depth_idx.append(min_time_diff_idx)

# create the 'synchro' folder
try:
	os.mkdir(os.path.join(DIR, 'synchro'))
except FileExistsError:
	print('The \'synchro\' folder already exists.')

depth_stamps_str = [str(depth_stamp).replace('.', '_') for depth_stamp in depth_stamps]	# doing the same operation forth
																						# and back, but still ok, no need
																						# to be super efficient
for idx in depth_idx:
	# create the a subfolder of 'synchro' named with the timestamp of the depth data in it
	current_dir = os.path.join(DIR, 'synchro', depth_stamps_str[idx])
	try:
		os.mkdir(current_dir)
	except FileExistsError:
		print('The ', depth_stamps_str[idx],' folder already exists.')

	shutil.copy(os.path.join(PCD_DIR, pcd_files[depth_idx.index(idx)]), current_dir)
	# the indexes found for the depth images can be applied to the RGB ones as they where
	# (approx) synchronously acquired
	shutil.copy(os.path.join(DEPTH_DIR, depth_files[idx]), current_dir)
	shutil.copy(os.path.join(RGB_DIR, rgb_files[idx]), current_dir)
	print('RGB/depth time diff.: ', abs(depth_stamps[idx] - rgb_stamps[idx]), '\n')	# TODO: check if the RGB and depth info
																					#		are too far in time
	shutil.copy(os.path.join(DEPTH_NORM_DIR, depth_norm_files[idx]), current_dir)