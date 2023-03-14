using LookMaNoHands
import LookMaNoHands: camera_warmup, snap

img = snap()

using FileIO

save("test.png", img)