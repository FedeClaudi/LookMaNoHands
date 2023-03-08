import GLMakie, VideoIO

cam = VideoIO.opencamera()
try
  img = read(cam)

  obs_img = GLMakie.Observable(GLMakie.rotr90(img))
  scene = GLMakie.Scene(camera=GLMakie.campixel!, resolution=reverse(size(img)))
  GLMakie.image!(scene, obs_img)
  display(scene)

#   fps = VideoIO.framerate(cam)
    fps = 30

  @info "setup complete" size(img) fps

  while GLMakie.isopen(scene)
    img = read(cam)
    obs_img[] = GLMakie.rotr90(img)
    sleep(1/fps)
  end
# catch e
#   @error e
finally
  close(cam)
end