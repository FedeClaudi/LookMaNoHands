module LookMaNoHands

import GLMakie, VideoIO
using FileIO
import ImageTransformations: imresize
using ImageFeatures, LIBSVM, Images

fps = 30
img_size = (360, 640)


include("camera.jl")
include("data/data.jl")

export stream_video, save_camera_frames

end
