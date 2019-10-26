import os
import re
import sys
from setuptools import setup, find_packages


PY36 = (3, 6, 0)
PY37 = (3, 7, 0)


if sys.version_info < PY36:
    raise RuntimeError('ibreakdown does not support Python earlier than 3.6')


install_requires = ['numpy', 'terminaltables']
if sys.version_info < PY37:
    install_requires.append('dataclasses==0.6')


def read(f):
    return open(os.path.join(os.path.dirname(__file__), f)).read().strip()


extras_require = {}


def read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(
        os.path.dirname(__file__), 'ibreakdown', '__init__.py'
    )
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        else:
            msg = 'Cannot find version in ibreakdown/__init__.py'
            raise RuntimeError(msg)


classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Operating System :: POSIX',
    'Development Status :: 2 - Pre-Alpha',
]


setup(
    name='ibreakdown',
    version=read_version(),
    description=('ibreakdown - model agnostic explanations with interactions'),
    long_description='\n\n'.join((read('README.rst'), read('CHANGES.txt'))),
    install_requires=install_requires,
    classifiers=classifiers,
    platforms=['POSIX'],
    author='Nikolay Novik',
    author_email='nickolainovik@gmail.com',
    url='https://github.com/jettify/ibreakdown',
    download_url='https://pypi.python.org/pypi/ibreakdown',
    license='Apache 2',
    packages=find_packages(),
    extras_require=extras_require,
    keywords=['ibreakdown', 'model explanation', 'xai'],
    zip_safe=True,
    include_package_data=True,
)
