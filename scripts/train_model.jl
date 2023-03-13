using LookMaNoHands
import LookMaNoHands: crop_images, load_features_data, data_types
using LIBSVM, FileIO, ImageFeatures, Images
import Plots

GENERATE_DATA = false
FIT_MODEL = true

gendata_tag = "face_null"
tag = "face"


# ---------------------------------------------------------------------------- #
#                                 GENERATE DATA                                #
# ---------------------------------------------------------------------------- #
GENERATE_DATA && crop_images(gendata_tag; width=data_types[gendata_tag].width)    


# ---------------------------------------------------------------------------- #
#                                     MODEL                                    #
# ---------------------------------------------------------------------------- #
FIT_MODEL && @time begin
    data, labels, descriptor_size = load_features_data(tag)

    # --------------------------------- fit model -------------------------------- #

    model = @time svmtrain(data, labels);
    descriptor = zeros(descriptor_size, 1)

    for (i, tag) in enumerate(("face", "face_null"))
        img = load("./imgs/$tag/1_0.png")
        descriptor[:, 1] = create_descriptor(img, HOG())

        predicted_label, _ = svmpredict(model, descriptor);
        println("Tag: $tag predicted_label: $predicted_label")    
    end
    # -------------------------- strided run on an image ------------------------- #

    img = load("./imgs/1.png")
    rows, cols = size(img)
    @info "loaded img" size(img) typeof(img) eltype(img)

    W = data_types[tag].width
    Δ = 25
    cols_iter = W:Δ:cols-W |> collect
    rows_iter =  W:Δ:rows-W |> collect
    scores = zeros(length(cols_iter), length(rows_iter))

    nsteps = length(cols_iter)
    for (k, j) in enumerate(cols_iter)
        for (m, i) in enumerate(rows_iter)
            box = view(img, i-W+1:i+W, j-W+1:j+W)
            descriptor[:, 1] = create_descriptor(box, HOG())
            predicted_label, s = svmpredict(model, descriptor);
            scores[k, m] = s[1]
        end
        println("Step: $k/$nsteps")
    end

    # println(scores)
    # Gray.(scores)
    Plots.heatmap(scores) |> display
    scores
end