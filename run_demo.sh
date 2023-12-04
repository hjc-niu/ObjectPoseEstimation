#!/bin/sh
# Execute the shell script from the build directory in the terminal
cd ..
./bin/object_pose_estimation_v1.0.0 ./pcd/model.pcd ./pcd/scene.pcd
