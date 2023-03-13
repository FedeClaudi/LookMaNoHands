function snap(cam)
    img = imresize(read(cam), img_size)
    return img 
end

function camera_warmup()
    cam = VideoIO.opencamera()
    snap(cam) # to make sure it works
    return cam
end

function stream_video()
    cam = camera_warmup()
    try
        img = snap(cam)

        obs_img = GLMakie.Observable(GLMakie.rotr90(img))
        scene = GLMakie.Scene(camera=GLMakie.campixel!, resolution=reverse(size(img)))
        GLMakie.image!(scene, obs_img)
        display(scene)

        while GLMakie.isopen(scene)
            img = snap(cam)
            obs_img[] = GLMakie.rotr90(img)
            sleep(1/fps)
        end
    finally
        close(cam)
    end
end

function save_camera_frames(dest::String, N::Int)
    @info "saving camera frames" dest N
    cam = camera_warmup()
    sleep(0.5)
    try
        for i in 1:N
            @info "Taking frame $i"
            sleep(0.5)
            img = snap(cam)
            save(joinpath(dest, "$(i).png"), img)
            sleep(0.5)
        end
    finally
        close(cam)
    end
end

function save_video(dest::String, videoname::String, n_frames::Int)
    cam = VideoIO.opencamera()
    img = snap(cam) # to make sure it works
    
    frames = []
    try
        for i in 1:n_frames
            i % 10 == 0 && @info "Taking frame $i"
            try
                img = snap(cam)
            catch
                @warn "Error while taking frame $i"
                return
            end
            push!(frames, img)
            sleep(1/fps)
        end
    finally
        close(cam)
    end
    
    VideoIO.save(
        joinpath(dest, "$(videoname).mp4"), 
        frames, framerate=fps, 
        encoder_options=(crf=24, preset="medium"), 
    )
end