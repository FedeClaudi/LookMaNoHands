function camera_warmup()
    cam = VideoIO.opencamera()
    read(cam) # to make sure it works
    return cam
end

function stream_video()
    cam = camera_warmup()
    try
        img = read(cam)

        obs_img = GLMakie.Observable(GLMakie.rotr90(img))
        scene = GLMakie.Scene(camera=GLMakie.campixel!, resolution=reverse(size(img)))
        GLMakie.image!(scene, obs_img)
        display(scene)

        while GLMakie.isopen(scene)
            img = read(cam)
            obs_img[] = GLMakie.rotr90(img)
            sleep(1/fps)
        end
    finally
        close(cam)
    end
end

function save_camera_frames(dest::String, N::Int; scale=1)
    @info "saving camera frames" dest N
    cam = camera_warmup()
    sleep(0.5)
    try
        for i in 1:N
            @info "Taking frame $i"
            sleep(0.5)
            img = imresize(read(cam), ratio=scale)
            save(joinpath(dest, "$(i).png"), img)
            sleep(0.5)
        end
    finally
        close(cam)
    end
end


