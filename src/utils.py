from os import listdir, makedirs
from os.path import isfile, isdir, join, exists, dirname


def file_listing(dir, extension=None):
    '''
    List all files (exclude dirs) in specified directory with (optional) given extension

    Args:
        dir (string): Director full path
        extension (string): (optional) Extension of files to filter
    '''
    files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    if extension:
        files = list(filter(lambda f: f.endswith('.' + extension), files))
    return files


def dir_listing(base_dir):
    '''
    List all subdirectories of given base dir

    Args:
        base_dir (string): Base directory to search
    '''
    return [join(base_dir, d) for d in listdir(base_dir) if isdir(join(base_dir, d))]


def mkdir(path):
    '''
    Make directory in specified path *recursively*

    Args:
        path (string): Directory path to create
    '''
    if not exists(path):
        makedirs(path)


def last_component(path):
    '''Get paths last component (split by `/`) including file extension'''
    return list(filter(None, path.split('/')))[-1]


def file_exists(path):
    '''Check whether path exists'''
    return isfile(path)


def relative_path(path):
    '''Get path relative to file itself instead of dir program was started in'''
    base_dir = dirname(__file__)
    return join(base_dir, path)


def get_file_name(filepath):
    '''Get file name - excluding file extension'''
    return last_component(filepath).split('.')[-2]


def take(gen, n):
    '''
    Take n elements from generator

    Args:
        gen (iterable): Generator instance
        n (int): Number of elements to take

    Returns:
        List of values from generator
    '''
    lst = []
    for _ in range(n):
        lst.append(next(gen))
    return lst
