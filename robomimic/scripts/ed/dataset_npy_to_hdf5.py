import argparse
import numpy as np
import h5py
import random

def dfs_dict(dicts, f: h5py.Group):
    for k, v in dicts.items():
        if isinstance(v, dict):
            if f.get(k, None) is None:
                f.create_group(k)
            # print(k, 'dict')
            dfs_dict(v, f[k])
        
        elif isinstance(v, list):
            if type(v[0]) == dict:
                f.create_group(k)
                for sub_k in v[0].keys():
                    if type(v[0][sub_k]) == np.ndarray or type(v[0][sub_k]) == int or type(v[0][sub_k]) == float:
                        if k=='obs':
                            data = np.stack([x[sub_k] for x in v])
                            if sub_k == 'actions':
                                B, L = data.shape
                                assert L % 10 == 0
                                data = data.reshape(B, 10, L // 10)
                                data = data[:, -1]
                            else:
                                data = data[:, -1]
                            f[k].create_dataset(sub_k, data=data)
                        else:
                            f[k].create_dataset(sub_k, data=np.stack([x[sub_k] for x in v]))
                        # breakpoint()
                    elif type(v[0][sub_k]) == dict:
                        f[k].create_group(sub_k)
                        for i in range(len(v)):
                            dfs_dict(v[i][sub_k], f[k][sub_k])
                    else:
                        raise NotImplementedError   
            elif type(v[0]) == np.ndarray or type(v[0]) == int or type(v[0]) == float:
                f.create_dataset(k, data=np.stack(v))
                
        else:
            # print(k, "reached")
            if type(v) == np.ndarray or type(v) == int or type(v) == float:
                value = v
            elif type(v) == str:
                value = v.encode('utf8')
            else:
                raise NotImplementedError 
                       
            if f.get(k, None) is None:
                f.create_dataset(k, data=value)
            else:
                tmp = f[k]
                del f[k]
                try: # for the first time
                    f[k] = np.stack([tmp, value])
                except: # for the following times
                    f[k] = np.concatenate([tmp, np.array([value])])
                    

def get_masks(data):
        
    # create data mask
    indices = list(range(len(data)))
    random.shuffle(indices)
    train_cnt = int(len(data) * 0.9)
    
    train_indices, valid_indices = indices[:train_cnt], indices[train_cnt:]
    train_demos = np.array(['demo_{}'.format(i).encode('utf8') for i in train_indices])
    valid_demos = np.array(['demo_{}'.format(i).encode('utf8') for i in valid_indices])
    return train_demos, valid_demos


def npy_to_hdf5(npy_path, hdf5_path):
    assert npy_path.endswith(".npy")
    assert hdf5_path.endswith(".hdf5")
    
    data = np.load(npy_path, allow_pickle=True)
    print("Loaded data from {}".format(npy_path))
    with h5py.File(hdf5_path, "w") as f:
        f.create_group('data')
        f.create_group('mask')

        train_demos, valid_demos = get_masks(data)
        f['mask'].create_dataset('train', data=train_demos)
        f['mask'].create_dataset('valid', data=valid_demos)
        
        
        # create data
        for i in range(len(data)):
            print("demo: {}/{}".format(i+1, len(data)), end='\r')
            f['data'].create_group('demo_{}'.format(i))
            current_data = data[i]
            dfs_dict(current_data, f['data/demo_{}'.format(i)])
    print(f"Saved {npy_path} to {hdf5_path}")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--npy", type=str, required=True, help="npy dataset path")
    arg_parser.add_argument("--hdf5", type=str, required=True, help="Output hdf5 path")
    
    args = arg_parser.parse_args()
    npy_to_hdf5(args.npy, args.hdf5)
    
