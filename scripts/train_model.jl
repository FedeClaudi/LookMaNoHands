using FileIO
using ImageFeatures, LIBSVM, Images

descriptor_size = 18972

# --------------------------------- load data -------------------------------- #

pos_examples = "./imgs/face"
neg_examples = "./imgs/noface"


n_pos = length(readdir(pos_examples))   # number of positive training examples
n_neg = length(readdir(neg_examples))   # number of negative training examples
n = n_pos + n_neg                       # number of training examples 
data = Array{Float64}(undef, descriptor_size, n)   # Array to store HOG descriptor of each image. Each image in our training data has size 128x64 and so has a 3780 length 
labels = Vector{Int}(undef, n)          # Vector to store label (1=human, 0=not human) of each image.

for (i, file) in enumerate([readdir(pos_examples); readdir(neg_examples)])
    filename = "$(i <= n_pos ? pos_examples : neg_examples )/$file"
    img = load(filename)
    
    i == 1 && @info "loaded img" size(img)  
    
    d = create_descriptor(img, HOG())
    data[:, i] = d
    labels[i] = (i <= n_pos ? 1 : 0)
end

@info "data ready" n_pos n_neg size(data) 

# --------------------------------- fit model -------------------------------- #

model = @time svmtrain(data, labels);

# -------------------------- strided run on an image ------------------------- #

img = load("./imgs/face/face_1.png")
rows, cols = size(img)

scores = zeros(22, 45)
descriptor = zeros(descriptor_size, 1)

for j in 32:10:cols-32
    for i in 64:10:rows-64
        box = img[i-63:i+64, j-31:j+32]
        @info "boxy" size(box)
        descriptor[:, 1] = create_descriptor(box, HOG())
        predicted_label, s = svmpredict(model, descriptor);
        scores[Int((i-64)/10)+1, Int((j-32)/10)+1] = s[1]
    end
end

display(imshow(scores))