using FileIO
using ImageFeatures, LIBSVM, Images

descriptor_size = 2916

# --------------------------------- load data -------------------------------- #
function load_data()
    eye_examples = "./imgs/eye"
    mouth_examples = "./imgs/mouth"
    null_examples = "./imgs/null"

    # get samples numbers
    n_eye = length(readdir(eye_examples))   # number of positive training examples
    n_mouth = length(readdir(mouth_examples))   # number of negative training examples
    n_null = length(readdir(null_examples))   # number of negative training examples
    
    min_n = min(n_eye, n_mouth, n_null)
    
    n = 3min_n                       # number of training examples

    # prepare data
    data = Array{Float64}(undef, descriptor_size, n)   # Array to store HOG descriptor of each image. Each image in our training data has size 128x64 and so has a 3780 length 
    labels = Vector{Int}(undef, n)          # Vector to store label (1=human, 0=not human) of each image.

    # load data and extract features
    count = 1
    for (lbl, fld) in enumerate([eye_examples, mouth_examples, null_examples])
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
    return data, labels
end

data, labels = load_data()
@info "data ready" size(data) 

# --------------------------------- fit model -------------------------------- #

model = @time svmtrain(data, labels);

for (i, tag) in enumerate(("eye", "mouth", "null"))
    img = load("./imgs/$tag/1_0.png")
    descriptor = zeros(descriptor_size, 1)
    descriptor[:, 1] = create_descriptor(img, HOG())

    predicted_label, _ = svmpredict(model, descriptor);
    println("Tag: $tag predicted_label: $predicted_label")    
end
# -------------------------- strided run on an image ------------------------- #

img = load("./imgs/1.png")
rows, cols = size(img)

scores = zeros(250, 250)

descriptor = zeros(descriptor_size, 1)

for j in 40:10:cols-40,  i in 40:10:rows-40
    println(i, " ", j)
    box = img[i-39:i+40, j-39:j+40]

    descriptor[:, 1] = create_descriptor(box, HOG())
    predicted_label, s = svmpredict(model, descriptor);

    î = Int((i-10)/10)+1
    ĵ = Int((j-10)/10)+1
    scores[, ] = s[1]
end

Gray.(scores)