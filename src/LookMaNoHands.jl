module LookMaNoHands

import GLMakie, VideoIO
using FileIO
import ImageTransformations: imresize

fps = 30

include("camera.jl")

export stream_video, save_camera_frames

end
