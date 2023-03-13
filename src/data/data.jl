data_types = Dict(
    "face" => (; :width => 160, :features => 54756,),
    "face_null"=> (; :width => 160, ),
    
    "eye" => (; :width => 32, :features => 3000),
    "mouth" => (; :width => 32, :features => 3000),
    "eye_null"=> (; :width => 32, :features => 3000),
)

include("features.jl")
include("generate_data.jl")