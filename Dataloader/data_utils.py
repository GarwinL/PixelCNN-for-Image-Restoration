'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import argparse
import functools
import os
import time
import math
import pickle
import sys
from numbers import Number

import numpy as np

#import gpustat

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import gc

import torchvision.transforms as transforms


from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset


# adapted from https://stackoverflow.com/questions/6632244/difference-in-a-dict
def dict_diff(left, right):
    left_only = set(left) - set(right)
    right_only = set(right) - set(left)
    different = {k for k in set(left) & set(right) if not left[k]==right[k]}
    return different, left_only, right_only

def parsed_args_to_obj(args):
    dargs = dict(vars(args))
    ks = list(dargs)
    for k in ks:
        parts = k.split(".")
        if len(parts) > 1:
            o = dargs
            for p in parts[:-1]:
                try:
                    o[p]
                except:
                    o[p] = {}
                o = o[p]
            o[parts[-1]] = dargs[k]
    return argparse.Namespace(**dargs)

def add_commandline_flag(parser, name_true, name_false, default):
    parser.add_argument(name_true, action="store_true", dest=name_true[2:])
    parser.add_argument(name_false, action="store_false", dest=name_true[2:])
    parser.set_defaults(**{name_true[2:]: default})

def add_commandline_networkparams(parser, name, features, depth, kernel, activation, bn):
    parser.add_argument("--{}.{}".format(name, "features"), type=int, default=features)
    parser.add_argument("--{}.{}".format(name, "depth"), type=int, default=depth)
    parser.add_argument("--{}.{}".format(name, "kernel"), type=int, default=kernel)
    parser.add_argument("--{}.{}".format(name, "activation"), default=activation)

    bnarg = "--{}.{}".format(name, "bn")
    nobnarg = "--{}.{}".format(name, "no-bn")
    add_commandline_flag(parser, bnarg, nobnarg, bn)

def check_expdir(expdir, basedir=None):
    if os.path.exists(expdir):
        return expdir

    expdir_concat = os.path.join(basedir, expdir)
    if os.path.exists(expdir_concat):
        return expdir_concat

    dirs = os.listdir(basedir)
    dirs = filter(lambda s: s.startswith(expdir), dirs)
    # if not len(dirs)==1:
    #     print("Ambiguous expdir {}, {}".format(expdir, len(dirs)))

    return os.path.join(basedir, dirs.__iter__().__next__())

def update_namespace(n1, n2):
    n1_d = dict(vars(n1))
    n1_d.update(dict(vars(n2)))
    return argparse.Namespace(**n1_d)


def get_result_dir(basedir, suffix):
    imax = 0
    dirs = os.listdir(basedir)
    for d in dirs:
        try:
            i = int(d[:4])
            if i >= imax:
                imax = i+1
        except: pass

    return "{:04d}-{}/".format(imax, suffix)

def save_script_call(filename, parsed_args):
    with open(filename, "wb") as f:
        pickle.dump({"argv": sys.argv, "parsed_args": parsed_args},f)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def none_grad(optimizer):
        """Clears the gradients of all optimized :class:`Variable` s."""
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = None

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


term_width = 200
try:
    _, term_width = os.popen('stty size', 'r').read().split()
except:
    pass
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current//total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s\t' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class RandomScale(object):
    def __init__(self, scaleRange, interpolation=Image.BILINEAR):
        self.scaleRange = scaleRange
        self.interpolation = interpolation

    def __call__(self,img):
        scale = np.random.randint(self.scaleRange[0], self.scaleRange[1])
        return transforms.Scale(scale, self.interpolation)(img)
def mymean(T,dims, keepdims=False):
    dims = sorted(dims)
    dims.reverse()
    m = functools.reduce(lambda T, dim: torch.mean(T, dim=dim, keepdim=keepdims), dims, T)
    return m

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

def accuracy(output, target, topk=(1,), size_average=True):
    """Computes the precision@k for the specified values of k"""
    maxk = min(max(topk), output.shape[1])
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).permute(1,0)

    res = []
    for k in topk:
        correct_k = correct[:,:k].float().sum(1, keepdim=True)
        if size_average:
            correct_k = correct_k.sum(0)  / batch_size
        res.append(correct_k.mul_(100.0))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, ema_decay=0.99, name=None):
        self.name = name
        self.reset()
        self.ema_decay = ema_decay

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.ema = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.ema == 0:
            self.ema = val
        else:
            self.ema = self.ema_decay * self.ema + (1-self.ema_decay) * val

        return self

class SubsampledDataset(Dataset):
    def __init__(self, base_dataset, idx=None, ratio=None):
        self.base_dataset = base_dataset
        if idx is not None:
            self.idx = idx
        else:
            n = len(base_dataset)
            k = np.floor(n * ratio).astype(np.int)
            self.idx = np.random.choice(n,(k,),replace=False)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        return self.base_dataset[self.idx[index]]

# =============================================================================
# def memusage(device=0):
#     gpu_stats = gpustat.GPUStatCollection.new_query()
#     item = gpu_stats.jsonify()["gpus"][device]
#     return item
# 
# def print_memory_info(module, input, output):
#     module_name = type(module).__name__
#     memstats = memusage(2)
#     # print(input)
#     # print(output)
#     print("{} \t {} \t {} \t {}".format(module_name, memstats["memory.used"], input[0].size(), output[0].size()))
# =============================================================================

def normalization_parameters(dataset):
    run_mean = 0
    run_sq = 0
    for x,_ in dataset:
        run_mean += x.mean(2).mean(1)
        run_sq += (x**2).mean(2).mean(1)

    mean = run_mean / len(dataset)
    sq = run_sq / len(dataset)

    std = (sq - mean**2)**0.5

    return mean.tolist(), std.tolist()

def get_module_name_dict(root, rootname="/"):
    def _rec(module, d, name):
        for key, child in module.__dict__["_modules"].items():
            d[child] = name + key + "/"
            _rec(child, d, d[child])

    d = {root: rootname}
    _rec(root, d, d[root])
    return d

def progressbar_string(stats, names_and_formatting, endofepoch=False):
    parts = []
    for n,f in names_and_formatting:
        if n in stats:
            s = stats[n]
            if endofepoch:
                parts.append(("%s: "+f)  %(s.name, s.avg))
            else:
                parts.append(("%s: "+f)  %(s.name, s.ema))
    return " | ".join(parts)

def update_stats(stats, name, val, count=1):
    if not name in stats:
        stats[name] = AverageMeter(name=name)
    stats[name].update(val, count)


def print_torch_tensors():
    l = gc.get_objects()
    for obj in l:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            if not obj.numel() > 10000*26000:
                continue
            volatile = obj.volatile if hasattr(obj, 'data') else True
            print(type(obj), obj.size(), volatile, [r.keys if isinstance(r, dict) else r for r in gc.get_referrers(obj) if r is not l])
            # print(type(obj), obj.size(), id(obj), volatile, [type(r) for r in gc.get_referrers(obj) if r is not l])

def distribution_stats(T):
    stats = {
        "min": T.cpu().min(),
        "max": T.cpu().max(),
        "median": T.cpu().median(),
        "mean": T.cpu().mean(),
        "std": T.cpu().std(),
    }
    return stats


def copy_module_hooks(mfrom, mto):
    mto._forward_pre_hooks = mfrom._forward_pre_hooks
    mto._forward_hooks = mfrom._forward_hooks
    mto._backward_hooks = mfrom._backward_hooks


def parameters_by_module(net, name):
    modulenames = get_module_name_dict(net, name + "/")
    params = [{"params": p, "name": n, "module": modulenames[m]} for m in net.modules() for n,p in m._parameters.items() if p is not None]
    return params

def parameter_count(modules):
    parameters = functools.reduce(lambda a,b:a+b, [parameters_by_module(m, str(i)) for i,m in enumerate(modules)])

    nparams = 0
    for pg in parameters:
        for p in pg["params"]:
            nparams+=p.data.numel()

    return nparams

def walklevel(some_dir, level=0):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level < num_sep_this and level >= 0:
            del dirs[:]

import contextlib
import sys

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout(out=True, err=True):
    if out:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
    if err:
        save_stderr = sys.stderr
        sys.stderr = DummyFile()
    yield
    if out:
        sys.stdout = save_stdout
    if err:
        sys.stderr = save_stderr

@contextlib.contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()
    fd2 = sys.stderr.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    def _redirect_stderr(to):
        sys.stderr.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stderr = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with os.fdopen(os.dup(fd2), 'w') as old_stderr:
            with open(to, 'w') as file:
                _redirect_stdout(to=file)
                _redirect_stderr(to=file)
            try:
                yield # allow code to be run with the redirected stdout
            finally:
                _redirect_stdout(to=old_stdout) # restore stdout.
                _redirect_stderr(to=old_stderr) #
                                            # buffering and flags such as
                                            # CLOEXEC may be different