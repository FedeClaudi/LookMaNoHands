import VideoIO
using Images
import LookMaNoHands: save_video

n_frames = 600

vidname = "vid3"
dest = "/Users/federicoclaudi/Desktop/lmnh/videos"

save_video(dest, vidname, n_frames)