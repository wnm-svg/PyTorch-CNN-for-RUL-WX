import numpy as np
from dataset_prepare import *



def test_validation(Xt_ss, max_cycle_t, nf):
    test_img_v = []
    for i in range(len(max_cycle_t)):
        img_t_v = Xt_ss[max_cycle_t[i]-15:max_cycle_t[i], ]
        img_t_v = img_t_v.astype('float32')
        test_img_v.append(img_t_v) 
        tmp = max_cycle_t[i] 
        i += 1
        if(i < len(max_cycle_t)):
            max_cycle_t[i] += tmp
    
    test_validation = np.array(test_img_v)
    test_validation = test_validation.reshape(test_validation.shape[0], 1, 15, nf)
    test_validation = torch.from_numpy(test_validation)
        
    return test_validation

if __name__ == "__main__":
    train_raw, test_raw, max_cycle, max_cycle_t, y_test = load_data_FD001()
    X_ss, idx, Xt_ss, idx_t, nf, ns, ns_t = get_info(train_raw, test_raw)
    # include last example of every engine , 100 examples totally
    validation = test_validation(Xt_ss, max_cycle_t, nf)
    print(len(validation))

