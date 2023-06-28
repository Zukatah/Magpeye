import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("Pictures3D_Input", output="Pictures3D",
    seed=1337, ratio=(.9, .1), group_prefix=None, move=True) # default values