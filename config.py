# =============================================================================
# model config GazeFollow360
# =============================================================================
dataset_root_path = "/home/data/tbw_gaze/training_dataset/gazefollow360/GazeFollow360_dataset"
# face_resolution = (128, 128)
# scene_resolution = (512, 256) 
# output_resolution = (72, 136)
face_resolution = (224, 224)
scene_resolution = (224, 224) 
output_resolution = (224, 224)
# is_jitter = True
# is_flip = True
# is_color_distortion = True
is_jitter = False
is_flip = False
is_color_distortion = False
gpuid = "2"