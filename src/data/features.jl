function load_features_data(tag)
    descriptor_size = data_types[tag].features
    face_data = "./imgs/face"
    null_data = "./imgs/face_null"

    # get samples numbers
    n_face = length(readdir(face_data))   # number of positive training examples
    n_null = length(readdir(null_data))   # number of negative training examples
    
    min_n = min(n_face, n_null)
    n = 2min_n                       # number of training examples

    # prepare data
    data = Array{Float64}(undef, descriptor_size, n)   # Array to store HOG descriptor of each image. Each image in our training data has size 128x64 and so has a 3780 length 
    labels = Vector{Int}(undef, n)          # Vector to store label (1=human, 0=not human) of each image.

    # load data and extract features
    count = 1
    for (lbl, fld) in enumerate([face_data, null_data])
        for (i, file) in enumerate(readdir(fld))
            i > min_n && break
            filename = "$(fld)/$file"
            img = load(filename)        
            d = create_descriptor(img, HOG())
            data[:, count] = d
            labels[count] = lbl
            count += 1
        end
    end

    return data, labels, descriptor_size
end