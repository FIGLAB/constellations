import numpy as np
from protocol import SensorType, SensorName, StatsFeature
import features
from features import EventInstance, DataInstances, reverse_map_location
import pandas as pd
import devices

def extract_flat_features(input, num_channels):
    N = input.shape[0]
    M = int(input.shape[1] / num_channels)
    output = np.zeros((N,M*2))

    for n in range(N):
        folded = input[n,0:].reshape(num_channels,-1)
        max_feats = np.max(folded,axis=0)
        mean_feats = np.mean(folded, axis=0)
        output[n,0:M] = max_feats
        output[n,M:] = mean_feats

    return output

def extract_flat_features_preserve_current_room(X, num_channels, room):

    keyset = list(X.keys())
    sorted(keyset)

    # Assemble
    N = X[keyset[0]].shape[0]
    C = conv_features_for_in_room(X, room)
    M = C.shape[1] * len(keyset)
    feats = np.zeros((N, M))

    # Make first channel = current room
    pos = 0
    stride = C.shape[1]
    feats[:,pos:pos+stride] = C
    pos += stride

    # Remove current room in keyset
    keyset.remove(room)

    for k in keyset:
        C = conv_features_for_in_room(X, k)
        stride = C.shape[1]
        feats[:, pos:pos + stride] = C
        pos += stride

    return feats

def extract_flat_features_preserve_current_room_and_adjacent_room(X, room):

    keyset = list(X.keys())
    sorted(keyset)

    # Grab Adjacent Rooms
    adjacent_rooms = devices.adjacencies[reverse_map_location(room)]

    # Assemble
    N = X[keyset[0]].shape[0]
    C = conv_features_for_in_room(X, room)
    M = C.shape[1] * (len(adjacent_rooms)+1)
    feats = np.zeros((N, M))

    # Make first channel = current room
    pos = 0
    stride = C.shape[1]
    feats[:,pos:pos+stride] = C
    pos += stride

    # Remove current room in keyset
    keyset.remove(room)

    for k in adjacent_rooms:
        C = conv_features_for_in_room(X, k)
        stride = C.shape[1]
        feats[:, pos:pos + stride] = C
        pos += stride

    input = feats
    num_channels = len(adjacent_rooms) + 1
    N = input.shape[0]
    M = int(input.shape[1] / num_channels)
    output = np.zeros((N,M*3))

    for n in range(N):
        output[n,0:M] = input[n,0:M]

    for n in range(N):
        folded = input[n,M:].reshape(num_channels-1,-1)
        max_feats = np.max(folded,axis=0)
        mean_feats = np.mean(folded, axis=0)
        output[n,M:M*2] = max_feats
        output[n,M*2:M*3] = mean_feats

    return output

def extract_flat_features_adjacent_rooms(X, room):

    # Grab Adjacent Rooms
    adjacent_rooms = devices.adjacencies[reverse_map_location(room)]

    # Assemble
    N = X[room].shape[0]
    C = conv_features_for_in_room(X, room)
    M = C.shape[1] * len(adjacent_rooms)
    feats = np.zeros((N, M))

    # Make first channel = current room
    pos = 0
    stride = C.shape[1]

    for k in adjacent_rooms:
        C = conv_features_for_in_room(X, k)
        stride = C.shape[1]
        feats[:, pos:pos + stride] = C
        pos += stride

    input = feats
    num_channels = len(adjacent_rooms)
    N = input.shape[0]
    M = int(input.shape[1] / num_channels)
    output = np.zeros((N,M*2))

    for n in range(N):
        folded = input[n,0:].reshape(num_channels,-1)
        max_feats = np.max(folded,axis=0)
        mean_feats = np.mean(folded, axis=0)
        output[n,0:M] = max_feats
        output[n,M:M*2] = mean_feats

    return output

def extract_flat_features_preserve_current_room_and_nearest_room(X, room):

    # Grab Adjacent Rooms
    remap = reverse_map_location(room)
    nearest_room = devices.nearest[remap]

    # Assemble
    N = X[room].shape[0]
    C = conv_features_for_in_room(X, room)
    M = C.shape[1] * (len(nearest_room)+1)
    feats = np.zeros((N, M))

    # Make first channel = current room
    pos = 0
    stride = C.shape[1]
    feats[:,pos:pos+stride] = C
    pos += stride

    for k in nearest_room:
        C = conv_features_for_in_room(X, reverse_map_location(k)) # conv_features_for_in_room(X, k)
        stride = C.shape[1]
        feats[:, pos:pos + stride] = C
        pos += stride

    return feats

def confusion_matrix(unique,res,Y_test):
    out = []
    per_sensor_acc = dict()
    target = np.array(Y_test)
    prediction = np.array(res)
    for i in range(len(unique)):
        out.append("%d = %s" % (i,unique[i]))
        indices = np.where(target==i)
        per_sensor_acc[unique[i]] = np.mean(target[indices]==prediction[indices])
    
    accuracy = np.mean(res==Y_test)
    
    t = Y_test
    p = res

    y_actu = pd.Series(t, name='Actual')
    y_pred = pd.Series(p, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, margins=True)
    df_confusion = "%s\n%s" % (", ".join(out),df_confusion)
    return accuracy,df_confusion,per_sensor_acc
        
def conv_features_for_in_room(X, room, normalize=False): # usually False
    # FFT Features:
    fft_feats = range(128*5)
    
    # GridEye (64*7)
    geye_feats = []
    for ch in range(64):
        for k in range(7):
            geye_feats.append(features.resolve(SensorType.GEYE, ch, k))
     
    # Single-Channel Sensors (9*7+1)
    temp_feats = []
    baro_feats = []
    hum_feats = []
    ir_feats = []
    illum_feats = []
    wifi_feats = []
    accel_stats = []
    mic_stats = []
    emi_stats = []
    for k in range(7):
        mic_stats.append(features.resolve(SensorType.MICROPHONE,0,k))
        emi_stats.append(features.resolve(SensorType.EMI,0,k))
        temp_feats.append(features.resolve(SensorType.TEMPERATURE,0,k))
        baro_feats.append(features.resolve(SensorType.BAROMETER,0,k))
        hum_feats.append(features.resolve(SensorType.HUMIDITY,0,k))
        ir_feats.append(features.resolve(SensorType.IRMOTION,0,k))
        illum_feats.append(features.resolve(SensorType.ILLUMINATION,0,k))
        wifi_feats.append(features.resolve(SensorType.WIFI_RSSI,0,k))
    
    # COLOR
    color_feats = []
    for ch in range(3):
        for k in range(7):
            color_feats.append(features.resolve(SensorType.COLOR,ch,k))
            accel_stats.append(features.resolve(SensorType.ACCEL,ch,k))
    
    # Assemble!
    all_feats = [fft_feats, geye_feats, color_feats, illum_feats, ir_feats, temp_feats, baro_feats, hum_feats, wifi_feats, accel_stats, mic_stats, emi_stats]
    indices = np.array([item for sublist in all_feats for item in sublist])
    feats = X[room][:,indices]

    # Normalize
    if (normalize):
        accel0 = range(128)
        accel1 = range(128,128*2)
        accel2 = range(128*2, 128 * 3)
        mic0 = range(128*3, 128 * 4)
        emi0 = range(128*4, 128 * 5)
        geye64 = range(128*5, 128*5 + 64)
        color0 = range(128 * 5 + 64, 128 * 5 + 64 + 7*1)
        color1 = range(128 * 5 + 64 + 7*1, 128 * 5 + 64 + 7*2)
        color2 = range(128 * 5 + 64 + 7*2, 128 * 5 + 64 + 7*3)
        illum0 = range(128 * 5 + 64 + 7*3, 128 * 5 + 64 + 7*4)
        ir0 = range(128 * 5 + 64 + 7*4, 128 * 5 + 64 + 7*5)
        ranges = [accel0, accel1, accel2, mic0, emi0, geye64, color0, color1, color2, illum0, ir0]

        for r in ranges:
            feats[:,r] = Norm(feats[:,r])

    return feats

def conv_features_for_all_rooms(X,rooms=None):
    # Sort keys
    keyset = list(X.keys()) if rooms is None else rooms
    sorted(keyset)
    
    # Assemble
    N = X[keyset[0]].shape[0]
    C = conv_features_for_in_room(X,keyset[0])
    M = C.shape[1] * len(keyset)
    feats = np.zeros((N,M))
    pos = 0
    for k in keyset:
        C = conv_features_for_in_room(X,k)
        stride = C.shape[1]
        feats[:,pos:pos+stride] = C
        pos += stride
    return feats

def conv_features_for_all_but_in_room(X,room):
    # Sort keys
    keyset = list(X.keys())
    keyset.remove(room)
    return conv_features_for_all_rooms(X,keyset)

def flat_features_for_in_room(X, room):
    return X[room]

def flat_features_for_all_rooms(X):
    # Sort keys
    keyset = list(X.keys())
    sorted(keyset)
    
    # Assemble
    N = X[keyset[0]].shape[0]
    M = X[keyset[0]].shape[1] * len(keyset)
    feats = np.zeros((N,M))
    for n in range(N):
        pos = 0
        for k in keyset:
            stride = X[k].shape[1]
            feats[n,pos:pos+stride] = X[k][n]
            pos += stride
    return feats

def flat_features_for_all_but_in_room(X, room):
    # Sort keys
    keyset = list(X.keys())
    keyset.remove(room)
    sorted(keyset)
    
    # Assemble
    N = X[keyset[0]].shape[0]
    M = X[keyset[0]].shape[1] * len(keyset)
    feats = np.zeros((N,M))
    for n in range(N):
        pos = 0
        for k in keyset:
            stride = X[k].shape[1]
            feats[n,pos:pos+stride] = X[k][n]
            pos += stride
    return feats

def one_hot(all_labels):
    # Create a one-hot encoded vector
    unique = np.sort(np.unique(all_labels)) # Sorted alphabetically
    label_mapper = dict()
    for i in range(len(unique)):
        k = unique[i]
        label_mapper[k] = i
    
    label_indices = np.array([label_mapper[k] for k in all_labels])
    n_values = np.max(len(unique))
    Y = np.eye(n_values)[label_indices]
    return Y

def encode_labels(all_labels):
    # Create an encoded label (non one-hot)
    unique = np.sort(np.unique(all_labels)) # Sorted alphabetically
    label_mapper = dict()
    for i in range(len(unique)):
        k = unique[i]
        label_mapper[k] = i
    
    label_indices = np.array([label_mapper[k] for k in all_labels])
    return label_indices

def filter_mask(arr, item):
    a = np.array(arr)
    mask = np.ones(len(arr), dtype=bool)
    m = np.where(a==item)
    mask[m] = False
    return mask

def replace(arr, item, replacement):
    for i in range(len(arr)):
        arr[i] = replacement  if arr[i]==item else arr[i]
    return arr

def create_subset(subset, dataset, method=None, room=None, feature_type='conv'):
    # Allocate Data
    num_set = 0
    for k in subset:
        num_set += len(dataset[k]['labels'])
    
    # Assemble Dataset
    X_set = None
    Y_set = None
    label_mapping = None
    
    for k in subset:
        # Generate Features
        if (method=='in-room'):
            X = conv_features_for_in_room(dataset[k]['data'],room)

        if (method=='nearest-room'):
            X = extract_flat_features_preserve_current_room_and_nearest_room(dataset[k]['data'], room)

        if (method=='nearest-room-only'):
            X = conv_features_for_in_room(dataset[k]['data'], devices.nearest[reverse_map_location(room)][0])

        if (method=='all-rooms'):
            X = extract_flat_features_preserve_current_room(dataset[k]['data'], 10, room)

        if (method=='all-but-in-room'):
            X = conv_features_for_all_but_in_room(dataset[k]['data'], room)
            X = extract_flat_features(X, 9)

        # Filter Noise
        Y = dataset[k]['labels']
        label_mapping = np.sort(np.unique(Y))
        #print(np.sort(np.unique(Y)))
        filter = True
        if (filter):
            m = filter_mask(Y,'noise')
            Y = np.array(Y)[m]
            X = X[m]

        # Encode Label
        Y_one_hot = one_hot(Y)
        X_set = np.concatenate((X_set,X),axis=0) if X_set is not None else X
        Y_set = np.concatenate((Y_set,Y_one_hot),axis=0) if Y_set is not None else Y_one_hot

    return X_set,Y_set,label_mapping


def Norm(X):
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    norm_a = np.array([xmax - xmin])
    norm_a[norm_a == 0] = 1  # prevent divide-by-zero
    norm_b = np.array([xmin])

    X = (X - norm_b) / norm_a
    return X
