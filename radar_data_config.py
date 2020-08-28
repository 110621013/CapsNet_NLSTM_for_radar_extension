point_num = (18-9+1)*(20-5+1) #160 points cover taiwan
train_ratio = 0.7
photo_num = 11213
train_num = int(photo_num*train_ratio)
test_num = photo_num-train_num
length, width = 28, 28