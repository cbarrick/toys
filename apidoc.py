import ast
from importlib import import_module
from itertools import groupby
from pathlib import Path
from types import ModuleType


def split(dotted):
    '''Split a dotted name at the final dot.
    '''
    return dotted.rsplit('.', 1)


def init_path(dotted):
    '''Converts a dotted package name into the path to ``__init__.py``.
    '''
    p = dotted.split('.')
    p = '/'.join(p)
    p = Path(p) / '__init__.py'
    return p


def readme_path(dotted):
    '''Converts a dotted package name into the path to ``README.rst``.
    '''
    p = dotted.split('.')
    p = '/'.join(p)
    p = Path(p) / 'README.rst'
    return p


def collect_exports(dotted):
    '''Returns a mapping from public name to private name for all
    exported objects in a package and its subpackages.
    '''
    x = {}
    p = init_path(dotted)
    code = p.read_text()
    root = ast.parse(code, p)

    for node in ast.iter_child_nodes(root):
        if not isinstance(node, ast.ImportFrom):
            continue

        if 0 < node.level:
            for alias in node.names:
                name = dotted + '.' + alias.name
                asname = dotted + '.' + (alias.asname or alias.name)
                if node.module is None:
                    assert asname == name
                    x.update(collect_exports(name))
                else:
                    realname = dotted + '.' + node.module + '.' + alias.name
                    x[asname] = realname

    return x


def groupby_parent(exports):
    '''Group exports by parent package.
    '''
    parent = lambda x: split(x[0])[0]
    groups = sorted(exports.items())
    groups = groupby(groups, key=parent)
    groups = {k:dict(v) for k,v in groups}
    return groups


def get(dotted):
    '''Import an object from its dotted name.
    '''
    parent, name = split(dotted)
    mod = import_module(parent)
    obj = getattr(mod, name)
    return obj


def entry(dotted):
    '''Create an autodoc entry for the object named by `dotted`.
    '''
    obj = get(dotted)
    if isinstance(obj, Exception):
        entry = f'.. autoexception:: {dotted}'
        entry += '\n   :members:'
        entry += '\n   :special-members:'

    elif isinstance(obj, type):
        entry = f'.. autoclass:: {dotted}'
        entry += '\n   :members:'
        entry += '\n   :special-members:'

    elif isinstance(obj, ModuleType):
        entry = f'.. automodule:: {dotted}'
        entry += '\n   :members:'

    elif callable(obj):
        entry = f'.. autofunction:: {dotted}'

    else:
        entry = f'.. autodata:: {dotted}'

    return entry


def main(root_package='toys', outdir='docs/api'):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    all_exports = collect_exports(root_package)

    for package, exports in groupby_parent(all_exports).items():
        doc_path = outdir / f'{package}.rst'
        readme = readme_path(package)

        with open(doc_path, 'w') as fd:
            if readme.exists():
                fd.write(readme.read())
            else:
                title = f'Package {package}'
                print('========' * 10, file=fd)
                print(f'{title:^80}', file=fd)
                print('========' * 10, file=fd)
                print('\n', file=fd)

            for public_name in exports:
                print(entry(public_name), file=fd)


if __name__ == '__main__':
    main()
