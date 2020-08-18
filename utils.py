import torch as t

def load_weight(path, remove_decoder=False, remove_encoder_adapter=False, shift_comb=False):
    import copy
    device = t.device('cpu')
    state_dict = t.load(path, map_location=device)
    try:optimizer = state_dict['optimizer']
    except:optimizer = None
    state_dict = state_dict['student_model']
    state_dict_v2 = copy.deepcopy(state_dict)
    for key in state_dict:
        if ('adaptation_layer.layers.' in key or 'adaptation_layer_e.layers.' in key) and 'conv' in key:
            n1, n2, n3, n4, n5 = key.split('.')
            new_key = '{}.{}.{}.{}.{}'.format(n1, n2, n3, n4[4:], n5)
            state_dict_v2[new_key] = state_dict_v2.pop(key)
        if remove_decoder:
            if 'dcer' in key:
                state_dict_v2.pop(key)
        if remove_encoder_adapter:
            if 'adaptation_layer_e' in key:
                state_dict_v2.pop(key)
        if shift_comb:
            if 'comb' in key:
                n1, n2, n3, n4, n5 = key.split('.')
                new_key = '{}.{}.{}.{}.{}'.format(n1, n2, n3, int(n4)+1, n5)
                state_dict_v2[new_key] = state_dict_v2.pop(key)
    #for key in state_dict_v2:
    #    print(key, state_dict_v2[key].shape)
    #xexit()
    return state_dict_v2, optimizer


def save_epoch(path, model, optimizer=None):
    d = {}
    d['student_model'] = model.student_net.state_dict()
    if optimizer:
        d['optimizer'] = optimizer.state_dict()
    t.save(d, path)

def save_weight(smallest_val, best_epoch, loss_val, epoch, direct, model_name,
                model, optimizer):
    if smallest_val is None:
        path = '{}/{}/{}_{:f}.pth'.format(direct, model_name, epoch, loss_val)
        save_epoch(path, model, optimizer)
        best_epoch = epoch
        smallest_val = loss_val
    else:
        if loss_val < smallest_val:
            path = '{}/{}/{}_{:f}.pth'.format(direct, model_name, epoch, loss_val)
            if epoch % 5 == 0:
                save_epoch(path, model, optimizer)
            else:
                save_epoch(path, model)
            best_epoch = epoch
            smallest_val = loss_val
    if epoch > 30:
        path = '{}/{}/{}_{:f}.pth'.format(direct, model_name, epoch, loss_val)
        if epoch % 10 == 0:
            save_epoch(path, model, optimizer)
    return smallest_val, best_epoch, model, optimizer
