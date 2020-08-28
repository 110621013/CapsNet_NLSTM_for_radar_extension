import os
import numpy as np
from PIL import Image
#from our_data_config import *
from radar_data_config import *

dataset = 'radar_data'

def move_img():
    from_path = 'C:/Users/user/Desktop/radar/'
    to_path = 'C:/Users/user/Desktop/all_conbine/img/'+dataset+'/'
    folder_list = os.listdir(from_path)
    for folder in folder_list:
        location_path = from_path+folder+'/'
        print(location_path)
        img_list = os.listdir(location_path)
        for img_name in img_list:
            try:
                img = Image.open(location_path+img_name).convert('L')
                img.save(to_path+img_name)
            except:
                print('oof:', img_name)

def move_radar_origin_img():
    from_path = 'C:/Users/user/Desktop/暑期實習所有code/summer_intern_downloaddata/905849c7-graphic-obsv.radar.small-tw.cwb-as.tw/'
    to_path = 'C:/Users/user/Desktop/all_conbine/img/radar_origin/'
    folder_list = os.listdir(from_path)
    for folder in folder_list:
        location_path = from_path+folder+'/'
        print(location_path)
        img_list = os.listdir(location_path)
        for img_name in img_list:
            try:
                img = Image.open(location_path+img_name).convert('L')
                img.save(to_path+img_name)
            except:
                print('oof:', img_name)

def delete_copy_img():
    to_path = 'C:/Users/user/Desktop/all_conbine/img/'+dataset+'/'
    os.chdir(to_path)
    img_list = os.listdir(to_path)
    for img_name in img_list:
        print(img_name)
        if img_name.split('.')[0][-1] == ')':
            os.system('del '+img_name)

def dot_plot():
    worded_map_img = 'C:/Users/user/Desktop/myself/original_dataset/training-images/1/202005261400.png'

    img = Image.open(worded_map_img).convert('L')
    img_array = np.array(img)
    print(img_array.shape)

    con=0
    with open('point.txt', 'r') as f:
        for line in f.readlines():
            con+=1
            x, y = line.strip().split(',')
            x, y = int(x), int(y)
            print(con, x, y)
            img_array[y, x] = 0
            img_array[y+1, x] = 0
            img_array[y, x+1] = 0
            img_array[y+1, x+1] = 0

    img = Image.fromarray(img_array)
    img.save('C:/Users/user/Desktop/all_conbine/img.png')

def make_our_data_npy():
    from_path = 'C:/Users/user/Desktop/all_conbine/img/'+dataset+'/'
    to_path = 'C:/Users/user/Desktop/all_conbine/DATA/'+dataset+'/'

    #抓所有圖資以及所有點位置
    img_list = os.listdir(from_path)
    all_point_location = parse_point()

    train_num = int(len(img_list)*train_ratio)
    train_img_array = np.empty((train_num, length, width), dtype=int)
    train_point_value_array = np.empty((train_num, point_num), dtype=int)
    test_img_array = np.empty((len(img_list)-train_num, length, width), dtype=int)
    test_point_value_array = np.empty((len(img_list)-train_num, point_num), dtype=int)

    #Image讀圖資再轉array, 且將所有不同圖片的點值跟圖片本身記錄下來
    for i in range(len(img_list)): #0~1386
        img_name = img_list[i]
        img = Image.open(from_path+img_name).convert('L')
        img_array = np.array(img)

        point_value = []
        for location in all_point_location:
            x, y = location[0], location[1]
            point_value.append(img_array[y, x])

        if i < train_num:
            train_point_value_array[i] = point_value
            train_img_array[i] = img_array
        else:
            test_point_value_array[i-train_num] = point_value
            test_img_array[i-train_num] = img_array

    np.save(to_path+'train_point_value_array', train_point_value_array)
    np.save(to_path+'train_img_array', train_img_array)
    np.save(to_path+'test_point_value_array', test_point_value_array)
    np.save(to_path+'test_img_array', test_img_array)

def parse_point():
    con = 0
    point = np.empty((point_num, 2), dtype=int)
    with open('point.txt', 'r') as f:
        for line in f.readlines():
            x, y = line.strip().split(',')
            x, y = int(x), int(y)
            point[con, 0] = x
            point[con, 1] = y
            con+=1
    return point


def make_radar_data_npy():
    from_path = 'C:/Users/user/Desktop/all_conbine/img/'+dataset+'/'
    to_path = 'C:/Users/user/Desktop/all_conbine/DATA/'+dataset+'/'
    img_list = os.listdir(from_path)
    all_point_location = point_range_of_taiwan()
    print(all_point_location)

    train_num = int(len(img_list)*train_ratio)
    train_img_array = np.empty((train_num, length, width), dtype=int)
    train_point_value_array = np.empty((train_num, point_num), dtype=int)
    test_img_array = np.empty((len(img_list)-train_num, length, width), dtype=int)
    test_point_value_array = np.empty((len(img_list)-train_num, point_num), dtype=int)

    #Image讀圖資再轉array, 且將所有不同圖片的點值跟圖片本身記錄下來
    for i in range(len(img_list)):
        print(i)
        img_name = img_list[i]
        img = Image.open(from_path+img_name).convert('L')
        img_array = np.array(img)

        point_value = []
        for location in all_point_location:
            x, y = location[0], location[1]
            point_value.append(img_array[y, x])

        if i < train_num:
            train_point_value_array[i] = point_value
            train_img_array[i] = img_array
        else:
            test_point_value_array[i-train_num] = point_value
            test_img_array[i-train_num] = img_array

    np.save(to_path+'train_point_value_array', train_point_value_array)
    np.save(to_path+'train_img_array', train_img_array)
    np.save(to_path+'test_point_value_array', test_point_value_array)
    np.save(to_path+'test_img_array', test_img_array)


def point_range_of_taiwan():
    point = np.empty((point_num, 2), dtype=int) #160
    con = 0
    for i in range(9, 18+1):
        for j in range(5, 20+1):
            point[con, 0] = i
            point[con, 1] = j
            con+=1
            #if con == point_num:print('okok', con, i, j)
    return point

def make_origin_radar_data_npy():
    edge = 70

    from_path = 'C:/Users/user/Desktop/all_conbine/img/radar_origin/'
    to_path = 'C:/Users/user/Desktop/all_conbine/DATA/'+dataset+'/'
    img_list = os.listdir(from_path)
    all_point_location = point_origin_range_of_taiwan()
    print(all_point_location, all_point_location.shape)

    train_num = int(len(img_list)*train_ratio)
    train_img_array = np.empty((train_num, edge, edge), dtype=int)
    train_point_value_array = np.empty((train_num, edge*edge), dtype=int)
    test_img_array = np.empty((len(img_list)-train_num, edge, edge), dtype=int)
    test_point_value_array = np.empty((len(img_list)-train_num, edge*edge), dtype=int)

    #Image讀圖資再轉array, 且將所有不同圖片的點值跟圖片本身記錄下來
    for i in range(len(img_list)):
        print(i)
        img_name = img_list[i]
        img = Image.open(from_path+img_name).convert('L')
        img_array = np.array(img)

        point_value = []
        for location in all_point_location:
            x, y = location[0], location[1]
            point_value.append(img_array[y, x])

        if i < train_num:
            train_point_value_array[i] = point_value
            train_img_array[i] = img_array
        else:
            test_point_value_array[i-train_num] = point_value
            test_img_array[i-train_num] = img_array

    np.save(to_path+'train_point_value_array_origin', train_point_value_array)
    np.save(to_path+'train_img_array_origin', train_img_array)
    np.save(to_path+'test_point_value_array_origin', test_point_value_array)
    np.save(to_path+'test_img_array_origin', test_img_array)


def point_origin_range_of_taiwan():
    edge = 70

    point = np.empty((edge*edge, 2), dtype=int)
    con = 0
    for i in range(edge):
        for j in range(edge):
            point[con, 0] = i
            point[con, 1] = j
            con+=1
            #if con == point_num:print('okok', con, i, j)
    return point

#--------------------------------------------------------
def make_sparse_npy():
    edge = 70
    sparse = 5
    point_per_edge = int(edge/sparse)

    from_path = 'C:/Users/user/Desktop/all_conbine/img/radar_origin/'
    to_path = 'C:/Users/user/Desktop/all_conbine/DATA/'+dataset+'/'
    img_list = os.listdir(from_path)
    point_location = point_sparse()
    print(point_location, point_location.shape)

    train_num = int(len(img_list)*train_ratio)
    train_img_array = np.empty((train_num, edge, edge), dtype=int)
    train_point_value_array = np.empty((train_num, point_per_edge*point_per_edge), dtype=int)
    test_img_array = np.empty((len(img_list)-train_num, edge, edge), dtype=int)
    test_point_value_array = np.empty((len(img_list)-train_num, point_per_edge*point_per_edge), dtype=int)

    #Image讀圖資再轉array, 且將所有不同圖片的點值跟圖片本身記錄下來
    for i in range(len(img_list)):
        print(i)
        img_name = img_list[i]
        img = Image.open(from_path+img_name).convert('L')
        img_array = np.array(img)

        point_value = []
        for location in point_location:
            x, y = location[0], location[1]
            point_value.append(img_array[y, x])

        if i < train_num:
            train_point_value_array[i] = point_value
            train_img_array[i] = img_array
        else:
            test_point_value_array[i-train_num] = point_value
            test_img_array[i-train_num] = img_array

    np.save(to_path+'train_point_value_sparse', train_point_value_array)
    np.save(to_path+'train_img_sparse', train_img_array)
    np.save(to_path+'test_point_value_sparse', test_point_value_array)
    np.save(to_path+'test_img_sparse', test_img_array)


def point_sparse():
    edge = 70
    sparse = 5
    point_per_edge = int(edge/sparse)

    point = np.empty((point_per_edge*point_per_edge, 2), dtype=int)
    con = 0
    for i in range(point_per_edge):
        for j in range(point_per_edge):
            point[con, 0] = i*sparse
            point[con, 1] = j*sparse
            con+=1
            #if con == point_num:print('okok', con, i, j)
    return point


if __name__ == '__main__':
    #move_img()
    #delete_copy_img()
    #dot_plot()
    #make_radar_data_npy()

    #move_radar_origin_img()
    #make_origin_radar_data_npy()
    make_sparse_npy()
